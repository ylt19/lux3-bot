from sys import stderr as err
from enum import IntEnum

import numpy as np
from collections import defaultdict

from .base import SPACE_SIZE, get_opposite, RELIC_REWARD_RANGE


class NodeType(IntEnum):
    unknown = -1
    empty = 0
    nebula = 1
    asteroid = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.type = NodeType.empty
        self.energy = 0

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    @property
    def relic(self):
        return self._relic

    @property
    def reward(self):
        return self._reward

    @property
    def explored_for_relic(self):
        return self._explored_for_relic

    @property
    def explored_for_reward(self):
        return self._explored_for_reward

    def update_relic_status(self, status: None | bool):
        if status is None:
            self._relic = False
            self._explored_for_relic = False
        else:
            self._relic = status
            self._explored_for_relic = True

    def update_reward_status(self, status: None | bool):
        if status is None:
            self._reward = False
            self._explored_for_reward = False
        else:
            self._reward = status
            self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self):
        return self.x, self.y

    def manhattan_distance(self, other: "Node") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


class Space:
    def __init__(self):
        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        self._relic_id_to_node = {}
        self._relic_nodes: set[Node] = set()
        self._reward_nodes: set[Node] = set()
        self._all_relics_found = False
        self._all_rewards_found = False
        self._reward_results = []

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def get_opposite_node(self, x, y) -> Node:
        return self.get_node(*get_opposite(x, y))

    def update(self, obs, team_to_reward=None):
        tile_type = obs["map_features"]["tile_type"]
        energy = obs["map_features"]["energy"]

        explored_new_nodes = False
        all_relics_found = True
        all_rewards_found = True
        for node in self:
            x, y = node.coordinates
            node.type = NodeType(int(tile_type[x][y]))
            node.energy = int(energy[x][y])

            if not node.explored_for_relic:
                if not node.is_unknown:
                    self._update_relic_status(x, y, status=False)
                    explored_new_nodes = True
                else:
                    all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        self._all_relics_found = all_relics_found
        self._all_rewards_found = all_rewards_found
        self._update_relic_status_from_observation(obs)

        if self.num_relics_found == len(obs["relic_nodes_mask"]):
            # all relics found, mark all nodes as explored for relics
            self._all_relics_found = True
            for node in self:
                if not node.explored_for_relic:
                    self._update_relic_status(*node.coordinates, status=False)
                    explored_new_nodes = True

        if explored_new_nodes:
            self._update_reward_status_from_relics_distribution()

        if team_to_reward:
            self._update_reward_results(obs, team_to_reward)

        self._update_reward_status_from_reward_results()

    def _update_relic_status_from_observation(self, obs):
        relic_nodes = obs["relic_nodes"]
        relic_nodes_mask = obs["relic_nodes_mask"]
        for relic_id, (mask, xy) in enumerate(zip(relic_nodes_mask, relic_nodes)):
            if mask and relic_id not in self._relic_id_to_node:
                x, y = int(xy[0]), int(xy[1])
                self._update_relic_status(x, y, status=True)
                self._relic_id_to_node[relic_id] = self.get_node(x, y)

    def _update_reward_status_from_relics_distribution(self):
        r = RELIC_REWARD_RANGE
        relic_map = np.zeros((SPACE_SIZE + 2 * r, SPACE_SIZE + 2 * r), np.int16)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y + r][node.x + r] = 1

        sub_shape = (r * 2 + 1, r * 2 + 1)
        view_shape = tuple(np.subtract(relic_map.shape, sub_shape) + 1) + sub_shape
        strides = relic_map.strides + relic_map.strides
        sub_matrices = np.lib.stride_tricks.as_strided(relic_map, view_shape, strides)
        reward_map = sub_matrices.sum(axis=(2, 3))

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_reward_results(self, obs, team_to_reward):
        for team_id, team_reward in team_to_reward.items():
            ship_nodes = set()
            for active, position in zip(
                obs["units_mask"][team_id], obs["units"]["position"][team_id]
            ):
                if active:
                    ship_nodes.add(self.get_node(*position))

            if not ship_nodes:
                continue

            self._reward_results.append({"nodes": ship_nodes, "reward": team_reward})
            # print(self._reward_results, file=err)

    def _update_reward_status_from_reward_results(self):
        filtered_results = []
        for result in self._reward_results:
            if result["reward"] == 0:
                for node in result["nodes"]:
                    self._update_reward_status(*node.coordinates, status=False)
                continue

            nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if n.explored_for_reward and not n.reward:
                    continue

                if n.reward:
                    known_reward += 1
                    continue

                nodes.add(n)

            if not nodes:
                continue

            reward = result["reward"] - known_reward

            if reward == 0:
                for node in nodes:
                    self._update_reward_status(*node.coordinates, status=False)
                continue

            if reward == len(nodes):
                for node in nodes:
                    self._update_reward_status(*node.coordinates, status=True)
                continue

            filtered_results.append({"nodes": nodes, "reward": reward})

        self._reward_results = filtered_results

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        opp_node = self.get_opposite_node(x, y)
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        opp_node = self.get_opposite_node(x, y)
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    @property
    def num_relics_found(self) -> int:
        return len(self._relic_nodes)

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    @property
    def all_relics_found(self) -> bool:
        return self._all_relics_found

    @property
    def all_rewards_found(self) -> bool:
        return self._all_rewards_found

    def show_map(self, ships=None):
        def int_to_str(i):
            s = str(int(i))
            return " " + s if len(s) < 2 else s

        ship_signs = (
            [" "] + [str(x) for x in range(1, 10)] + ["A", "B", "C", "D", "E", "F", "H"]
        )

        coordinates_to_num_ships = defaultdict(int)
        if ships:
            for ship in ships:
                coordinates_to_num_ships[ship.node.coordinates] += 1

        line = " + " + " ".join([int_to_str(x) for x in range(SPACE_SIZE)]) + "  +\n"
        str_grid = line
        for y, row in enumerate(self._nodes):

            str_row = []

            for x in row:
                if x.type == NodeType.unknown:
                    str_row.append("..")
                    continue

                if x.type == NodeType.nebula:
                    s1 = "ñ" if x.relic else "n"
                elif x.type == NodeType.asteroid:
                    s1 = "ã" if x.relic else "a"
                else:
                    s1 = "~" if x.relic else " "

                num_ships = coordinates_to_num_ships.get(x.coordinates, 0)
                s2 = ship_signs[num_ships]

                str_row.append(s1 + s2)

            str_grid += " ".join(
                [int_to_str(y), " ".join(str_row), int_to_str(y), "\n"]
            )

        str_grid += line
        print(str_grid, file=err)

    def show_energy_field(self):
        def int_to_str(i):
            s = str(int(i))
            return " " + s if len(s) < 2 else s

        line = " + " + " ".join([int_to_str(x) for x in range(SPACE_SIZE)]) + "  +\n"
        str_grid = line
        for y, row in enumerate(self._nodes):

            str_row = []
            for x in row:
                if x.type == NodeType.unknown:
                    str_row.append("..")
                else:
                    str_row.append(int_to_str(x.energy))

            str_grid += " ".join(
                [int_to_str(y), " ".join(str_row), int_to_str(y), "\n"]
            )

        str_grid += line
        print(str_grid, file=err)

    def show_exploration_info(self):
        def int_to_str(i):
            s = str(int(i))
            return " " + s if len(s) < 2 else s

        line = " + " + " ".join([int_to_str(x) for x in range(SPACE_SIZE)]) + "  +\n"
        str_grid = line
        for y, row in enumerate(self._nodes):

            str_row = []
            for x in row:
                if not x.explored_for_relic:
                    s1 = "."
                else:
                    s1 = "R" if x.relic else " "

                if not x.explored_for_reward:
                    s2 = "."
                else:
                    s2 = "P" if x.reward else " "

                str_row.append(s1 + s2)

            str_grid += " ".join(
                [int_to_str(y), " ".join(str_row), int_to_str(y), "\n"]
            )

        str_grid += line
        print(str_grid, file=err)
