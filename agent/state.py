import copy
import numpy as np
from sys import stderr as err
from collections import defaultdict

from .base import Params, get_spawn_location, chebyshev_distance
from .path import ActionType
from .space import Space, NodeType
from .fleet import Fleet


class State:
    def __init__(self, team_id):
        self.team_id = team_id
        self.global_step = 0  # global step of the game
        self.match_step = 0  # current step in the match
        self.match_number = 0  # current match number

        # visible space, not hidden by the fog of war
        self.space = Space()

        # explored space, regardless of whether it's hidden or not
        self.explored_space = Space()

        self.fleet = Fleet(team_id)
        self.opp_fleet = Fleet(1 - team_id)

        self._obstacles_movement_status = []

    def update(self, obs):
        if obs["steps"] > 0:
            self._update_step_counters()

        assert obs["steps"] == self.global_step
        assert obs["match_steps"] == self.match_step

        if self.match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self._update_explored_space()
            return

        points = int(obs["team_points"][self.team_id])
        reward = max(0, points - self.fleet.points)

        self.space.update(obs, team_to_reward={self.team_id: reward})
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        if (
            Params.OBSTACLE_MOVEMENT_PERIOD == 0
            or (self.global_step - 1) % Params.OBSTACLE_MOVEMENT_PERIOD != 0
        ):
            self.space.update_nodes_by_expected_sensor_mask(
                self.fleet.expected_sensor_mask()
            )

        self._update_explored_space()

    def _update_step_counters(self):
        self.global_step += 1
        self.match_step += 1
        if self.match_step > Params.MAX_STEPS_IN_MATCH:
            self.match_step = 0
            self.match_number += 1

    def _update_explored_space(self):
        self._update_explored_relics()
        self._update_explored_energy()
        self._update_explored_map()

    def _update_explored_map(self):
        if (
            Params.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Params.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Params.OBSTACLE_MOVEMENT_PERIOD > 0
            and (self.global_step - 1) % Params.OBSTACLE_MOVEMENT_PERIOD == 0
        ):
            self.explored_space.move(*Params.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

        obstacles_shifted = None
        for e, v in zip(self.explored_space, self.space):
            if not e.is_unknown and not v.is_unknown:
                if e.type != v.type:
                    obstacles_shifted = True
                elif obstacles_shifted is None and (
                    v.type == NodeType.asteroid or v.type == NodeType.nebula
                ):
                    obstacles_shifted = False

        self._obstacles_movement_status.append(obstacles_shifted)
        if not Params.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = _get_obstacle_movement_period(self._obstacles_movement_status)
            if period is not None:
                Params.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                Params.OBSTACLE_MOVEMENT_PERIOD = period
                print(
                    f"Find param OBSTACLE_MOVEMENT_PERIOD_FOUND = {period}",
                    file=err,
                )
                if period == 0:
                    direction = (0, 0)
                    Params.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                    Params.OBSTACLE_MOVEMENT_DIRECTION = direction
                    print(
                        f"Find param OBSTACLE_MOVEMENT_DIRECTION = {direction}",
                        file=err,
                    )

        if obstacles_shifted and not Params.OBSTACLE_MOVEMENT_DIRECTION_FOUND:
            direction = _get_obstacle_movement_direction(
                self.explored_space, self.space
            )
            if direction:
                Params.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                Params.OBSTACLE_MOVEMENT_DIRECTION = direction
                print(
                    f"Find param OBSTACLE_MOVEMENT_DIRECTION = {direction}",
                    file=err,
                )

                self.explored_space.move(
                    *Params.OBSTACLE_MOVEMENT_DIRECTION, inplace=True
                )
                obstacles_shifted = False
            else:
                print("WARNING: Can't find OBSTACLE_MOVEMENT_DIRECTION", file=err)

        if obstacles_shifted:
            if (
                Params.OBSTACLE_MOVEMENT_PERIOD_FOUND
                and Params.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            ):
                print("WARNING: OBSTACLE_MOVEMENTS params are incorrect", file=err)

            for e, v in zip(self.explored_space, self.space):
                e.type = v.type
        else:
            for e, v in zip(self.explored_space, self.space):
                if e.is_unknown and not v.is_unknown:
                    e.type = v.type

    def _update_explored_energy(self):
        energy_nodes_shifted = False
        for e, v in zip(self.explored_space, self.space):
            if e.energy is not None and v.energy is not None:
                if e.energy != v.energy:
                    energy_nodes_shifted = True

        if energy_nodes_shifted:
            for e, v in zip(self.explored_space, self.space):
                e.energy = v.energy
        else:
            for e, v in zip(self.explored_space, self.space):
                if e.energy is None and v.energy is not None:
                    e.energy = v.energy

    def _update_explored_relics(self):
        for relic_node in self.space.relic_nodes:
            node = self.explored_space.get_node(*relic_node.coordinates)
            node.update_relic_status(True)

        for reward_node in self.space.reward_nodes:
            node = self.explored_space.get_node(*reward_node.coordinates)
            node.update_reward_status(True)

    def steps_left_in_match(self) -> int:
        return Params.MAX_STEPS_IN_MATCH - self.match_step

    def copy(self) -> "State":
        return copy.deepcopy(self)

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        spawn_pointer = self._get_spawn_pointer()
        if spawn_pointer:
            actions[:] = (ActionType.sap, *spawn_pointer)

        for i, ship in enumerate(ships):
            if ship.node is not None:
                actions[i][0] = ActionType.center

            if ship.action_queue:
                a = ship.action_queue[0]
                actions[i] = (a.type.value, a.dx, a.dy)

        return actions

    def _get_spawn_pointer(self):
        spawn_location = get_spawn_location(self.team_id)
        target = None
        min_distance = 0
        for ship in self.fleet:
            next_position = ship.next_position()

            if next_position == spawn_location:
                continue

            distance = chebyshev_distance(spawn_location, next_position)
            if distance > Params.UNIT_SAP_RANGE:
                continue

            if target is None or min_distance > distance:
                min_distance = distance
                target = next_position

        if target:
            return target[0] - spawn_location[0], target[1] - spawn_location[1]

    def show_visible_map(self):
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_visible_energy_field(self):
        show_energy_field(self.space)

    def show_explored_map(self):
        show_map(self.explored_space, self.fleet, self.opp_fleet, hide_by_energy=False)

    def show_explored_energy_field(self):
        show_energy_field(self.explored_space)

    def show_exploration_info(self):
        show_exploration_info(self.space)

    def show_tasks(self):
        print("Tasks:", file=err)
        for ship in self.fleet:
            print(f" - {ship} : {ship.task}", file=err)


def show_map(space, my_fleet=None, opp_fleet=None, hide_by_energy=True):
    def int_to_str(i):
        s = str(int(i))
        return " " + s if len(s) < 2 else s

    ship_signs = (
        [" "] + [str(x) for x in range(1, 10)] + ["A", "B", "C", "D", "E", "F", "H"]
    )
    red_color = "\033[91m"
    blue_color = "\033[94m"
    yellow_color = "\033[93m"
    endc = "\033[0m"

    my_ships = defaultdict(int)
    for ship in my_fleet or []:
        my_ships[ship.node.coordinates] += 1

    opp_ships = defaultdict(int)
    for ship in opp_fleet or []:
        opp_ships[ship.node.coordinates] += 1

    line = " + " + " ".join([int_to_str(x) for x in range(Params.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Params.SPACE_SIZE):

        str_row = []

        for x in range(Params.SPACE_SIZE):
            node = space.get_node(x, y)

            if hide_by_energy:
                if node.energy is None:
                    str_row.append("..")
                    continue
            else:
                if node.type == NodeType.unknown:
                    str_row.append("..")
                    continue

            if node.type == NodeType.nebula:
                s1 = "ñ" if node.relic else "n"
            elif node.type == NodeType.asteroid:
                s1 = "ã" if node.relic else "a"
            else:
                s1 = "~" if node.relic else " "

            if node.reward:
                if s1 == " ":
                    s1 = "_"
                s1 = f"{yellow_color}{s1}{endc}"

            if node.coordinates in my_ships:
                num_ships = my_ships[node.coordinates]
                s2 = f"{blue_color}{ship_signs[num_ships]}{endc}"
            elif node.coordinates in opp_ships:
                num_ships = opp_ships[node.coordinates]
                s2 = f"{red_color}{ship_signs[num_ships]}{endc}"
            else:
                s2 = " "

            str_row.append(s1 + s2)

        str_grid += " ".join([int_to_str(y), " ".join(str_row), int_to_str(y), "\n"])

    str_grid += line
    print(str_grid, file=err)


def show_energy_field(space):
    def int_to_str(i):
        s = str(int(i))
        return " " + s if len(s) < 2 else s

    line = " + " + " ".join([int_to_str(x) for x in range(Params.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Params.SPACE_SIZE):

        str_row = []

        for x in range(Params.SPACE_SIZE):
            node = space.get_node(x, y)
            if node.energy is None:
                str_row.append("..")
            else:
                str_row.append(int_to_str(node.energy))

        str_grid += " ".join([int_to_str(y), " ".join(str_row), int_to_str(y), "\n"])

    str_grid += line
    print(str_grid, file=err)


def show_exploration_info(space):
    print(
        f"all relics found: {Params.ALL_RELICS_FOUND}, "
        f"all rewards found: {Params.ALL_REWARDS_FOUND}",
        file=err,
    )

    def int_to_str(i):
        s = str(int(i))
        return " " + s if len(s) < 2 else s

    line = " + " + " ".join([int_to_str(x) for x in range(Params.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Params.SPACE_SIZE):

        str_row = []

        for x in range(Params.SPACE_SIZE):
            node = space.get_node(x, y)
            if not node.explored_for_relic:
                s1 = "."
            else:
                s1 = "R" if node.relic else " "

            if not node.explored_for_reward:
                s2 = "."
            else:
                s2 = "P" if node.reward else " "

            str_row.append(s1 + s2)

        str_grid += " ".join([int_to_str(y), " ".join(str_row), int_to_str(y), "\n"])

    str_grid += line
    print(str_grid, file=err)


def _get_obstacle_movement_direction(space, next_space):
    for direction in [(1, -1), (-1, 1)]:
        moved_space = space.move(*direction, inplace=False)
        # show_map(moved_space, hide_by_energy=False)
        # show_map(next_space, hide_by_energy=False)

        match = True
        for n1, n2 in zip(moved_space, next_space):
            if not n1.is_unknown and not n2.is_unknown and n1.type != n2.type:
                match = False
                break

        if match:
            return direction


def _get_obstacle_movement_period(obstacles_movement_status):
    if not obstacles_movement_status:
        return

    if obstacles_movement_status[-1] is True:
        if len(obstacles_movement_status) - 21 % 40 < 20:
            return 20
        else:
            return 40
