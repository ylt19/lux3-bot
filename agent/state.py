import copy
import numpy as np
from sys import stderr as err
from collections import defaultdict

from .base import Params
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

    def update(self, obs):
        if obs["steps"] > 0:
            self._update_step_counters()

        assert obs["steps"] == self.global_step
        assert obs["match_steps"] == self.match_step

        if self.match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            return

        points = int(obs["team_points"][self.team_id])
        reward = max(0, points - self.fleet.points)

        self.space.update(obs, team_to_reward={self.team_id: reward})
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)
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
        energy_nodes_shifted = False
        obstacles_shifted = False

        for v in self.space:

            e = self.explored_space.get_node(*v.coordinates)

            if v.energy is not None:
                if e.energy is None:
                    e.energy = v.energy
                elif e.energy != v.energy:
                    energy_nodes_shifted = True

            if not v.is_unknown:
                if e.is_unknown:
                    e.type = v.type
                elif e.type != v.type:
                    obstacles_shifted = True

        if energy_nodes_shifted:
            for e in self.explored_space:
                e.energy = self.space.get_node(*e.coordinates).energy

        if obstacles_shifted:
            for e in self.explored_space:
                e.type = self.space.get_node(*e.coordinates).type

    def steps_left_in_match(self) -> int:
        return Params.MAX_STEPS_IN_MATCH - self.match_step

    def copy(self) -> "State":
        return copy.deepcopy(self)

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)
        for i, ship in enumerate(ships):
            if ship.action_queue:
                a = ship.action_queue[0]
                actions[i] = (a.type.value, a.dx, a.dy)
        return actions

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
