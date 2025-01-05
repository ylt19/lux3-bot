import numpy as np
from collections import defaultdict

from .path import NodeType
from .base import log, Global, SPACE_SIZE, Colors, nearby_positions


class Field:
    def __init__(self, state):
        self._state = state

        self._asteroid = None
        self._nebulae = None
        self._energy = None
        self._energy_gain = None
        self._vision = None
        self._opp_vision = None
        self._distance = None

    def clear(self):
        self._asteroid = None
        self._nebulae = None
        self._energy = None
        self._energy_gain = None
        self._vision = None
        self._opp_vision = None
        self._distance = None

    @property
    def space(self):
        return self._state.space

    @property
    def asteroid(self):
        if self._asteroid is None:
            self._update_space_fields()
        return self._asteroid

    @property
    def nebulae(self):
        if self._nebulae is None:
            self._update_space_fields()
        return self._nebulae

    @property
    def energy(self):
        if self._energy is None:
            self._update_space_fields()
        return self._energy

    @property
    def energy_gain(self):
        if self._energy_gain is None:
            self._update_space_fields()
        return self._energy_gain

    @property
    def vision(self):
        if self._vision is None:
            self._vision = self._create_vision_field()
        return self._vision

    @property
    def opp_vision(self):
        if self._opp_vision is None:
            self._opp_vision = self._create_opp_vision_field()
        return self._opp_vision

    @property
    def distance(self):
        if self._distance is None:
            self._distance = self._create_distance_field()
        return self._distance

    def _update_space_fields(self):
        self._asteroid, self._nebulae, self._energy, self._energy_gain = (
            self._create_space_fields()
        )

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

    def _create_vision_field(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
        for node in self.space:
            if node.is_visible:
                field[node.y, node.x] = 1
        return field

    def _create_opp_vision_field(self):
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

    def _create_distance_field(self):
        field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)

        for x in range(SPACE_SIZE):
            for y in range(SPACE_SIZE):
                field[y, x] = self._state.fleet.spawn_distance(x, y)

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
