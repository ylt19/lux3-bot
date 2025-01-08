import numpy as np
from functools import cached_property
from collections import defaultdict
from pathfinding import Grid, ResumableDijkstra
from scipy.signal import convolve2d

from .path import NodeType
from .base import log, Global, SPACE_SIZE, Colors, nearby_positions


class Field:
    def __init__(self, state):
        self._state = state

        self.asteroid, self.nebulae, self.energy, self.energy_gain = (
            self._create_space_fields()
        )

    @property
    def space(self):
        return self._state.space

    def _create_space_fields(self):
        asteroid_field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        nebulae_field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        energy_field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        energy_field[:] = Global.HIDDEN_NODE_ENERGY
        energy_gain_field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        for node in self.space:
            x, y = node.coordinates
            if node.type == NodeType.asteroid:
                asteroid_field[y, x] = 1
            elif node.type == NodeType.nebula:
                nebulae_field[y, x] = 1
            if node.energy is not None:
                energy_field[y, x] = node.energy
            energy_gain_field[y, x] = node.energy_gain
        return asteroid_field, nebulae_field, energy_field, energy_gain_field

    @cached_property
    def vision(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        for node in self.space:
            if node.is_visible:
                field[node.y, node.x] = 1
        return field

    @cached_property
    def opp_vision(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        field[np.where(self._state.opp_fleet.vision > 0)] = 1

        num_opp_ships_with_rewards, opp_reward_nodes = prob_opp_on_rewards(self._state)
        if num_opp_ships_with_rewards and opp_reward_nodes:
            prob = num_opp_ships_with_rewards / len(opp_reward_nodes)
            if prob == 1:
                for node in opp_reward_nodes:
                    for x, y in nearby_positions(
                        *node.coordinates, Global.UNIT_SENSOR_RANGE
                    ):
                        field[y, x] = 1
            else:
                node_to_num_probs = defaultdict(int)
                for node in opp_reward_nodes:
                    for x, y in nearby_positions(
                        *node.coordinates, Global.UNIT_SENSOR_RANGE
                    ):
                        node_to_num_probs[(x, y)] += 1

                num_rewards_without_ships = (
                    len(opp_reward_nodes) - num_opp_ships_with_rewards
                )
                for (x, y), num_probs in node_to_num_probs.items():
                    if num_probs > num_rewards_without_ships:
                        field[y, x] = 1
                    else:
                        no_vision_prob = (1 - prob) ** num_probs
                        field[y, x] = max(field[y, x], 1 - no_vision_prob)

        return field

    @cached_property
    def distance(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)

        for x in range(SPACE_SIZE):
            for y in range(SPACE_SIZE):
                field[y, x] = self._state.fleet.spawn_distance(x, y)

        return field

    @cached_property
    def opp_distance(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)

        for x in range(SPACE_SIZE):
            for y in range(SPACE_SIZE):
                field[y, x] = self._state.opp_fleet.spawn_distance(x, y)

        return field

    @cached_property
    def rear(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)

        visibility_weights = np.ones((SPACE_SIZE, SPACE_SIZE), np.float32)
        straight_weights = np.ones((SPACE_SIZE, SPACE_SIZE), np.float32)
        for node in self.space:
            if not node.is_walkable:
                visibility_weights[node.y, node.x] = -1
                straight_weights[node.y, node.x] = -1
            if node.is_visible:
                visibility_weights[node.y, node.x] = -1

        visibility_grid = Grid(visibility_weights)
        straight_grid = Grid(straight_weights)

        opp_spawn_position = self._state.opp_fleet.spawn_position
        visibility_rs = ResumableDijkstra(visibility_grid, opp_spawn_position)
        straight_rs = ResumableDijkstra(straight_grid, opp_spawn_position)

        for node in self.space:
            visibility_path = visibility_rs.find_path(node.coordinates)
            straight_path = straight_rs.find_path(node.coordinates)

            if len(straight_path) < len(visibility_path):
                field[node.y, node.x] = 1

        return field

    @cached_property
    def control(self):
        field = np.logical_or(self.vision > 0, self.rear > 0)
        field = np.logical_or(field, self.distance < SPACE_SIZE / 2)
        return field

    @cached_property
    def opp_sap_ships_potential_positions(self):
        out_of_control = np.logical_not(self.control)

        # is it possible for opponent's ships to reach this position
        field = np.logical_and(
            out_of_control, self.opp_distance <= self._state.match_step
        )

        # add opponent's ships that can sap
        for ship in self._state.opp_fleet:
            if ship.can_sap():
                x, y = ship.coordinates
                field[y, x] = 1

        return field

    @cached_property
    def possible_targets_for_opp_sap_ships(self):
        r = Global.UNIT_SAP_RANGE * 2 + 1
        sap_kernel = np.ones((r, r), dtype=np.float32)
        field = convolve2d(
            self.opp_sap_ships_potential_positions,
            sap_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        field = field > 0
        return field

    @cached_property
    def direct_targets_for_opp_sap_ships(self):
        ships_in_opp_vision = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        for ship in self._state.fleet:
            x, y = ship.coordinates
            if (
                ship.energy >= 0
                and self.opp_vision[y, x] > 0
                and self.possible_targets_for_opp_sap_ships[y, x] > 0
            ):
                ships_in_opp_vision[y, x] = 1

        sap_kernel = np.ones((3, 3), dtype=np.float32)
        field = convolve2d(
            ships_in_opp_vision,
            sap_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        field = field > 0
        return field

    @cached_property
    def vision_gain(self):
        r = 2 * Global.UNIT_SENSOR_RANGE + 1
        vision_kernel = np.ones((r, r), dtype=np.float32)

        field = convolve2d(
            (self.control == 0),
            vision_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        return field


def show_field(weights):
    def add_color(i):
        color = Colors.green if i > 0 else Colors.red
        return f"{color}{i:>3}{Colors.endc}"

    line = " + " + " ".join([f"{x:>2}" for x in range(Global.SPACE_SIZE)]) + "  +\n"
    str_grid = line
    for y in range(Global.SPACE_SIZE):

        str_row = []

        for x in range(Global.SPACE_SIZE):
            v = int(weights[y, x])
            str_row.append(add_color(v))

        str_grid += "".join([f"{y:>2}", *str_row, f" {y:>2}", "\n"])

    str_grid += line
    log(str_grid)


def prob_opp_on_rewards(state):
    if not Global.ALL_REWARDS_FOUND:
        return 0, set()

    reward_nodes = set()
    for reward_node in state.space.reward_nodes:
        if not reward_node.is_visible:
            reward_nodes.add(reward_node)

    opp_rewards_in_vision = 0
    for ship in state.opp_fleet:
        if ship.node.reward:
            opp_rewards_in_vision = +1

    num_opp_ships_with_rewards = state.opp_fleet.reward - opp_rewards_in_vision

    return num_opp_ships_with_rewards, reward_nodes
