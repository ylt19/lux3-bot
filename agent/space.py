import numpy as np
from sys import stderr as err
from enum import IntEnum

from .base import Params, SPACE_SIZE, get_opposite


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
        self.type = NodeType.unknown
        self.energy = None

        self._relic = False
        self._reward = False
        self._explored_for_relic = False
        self._explored_for_reward = False

    def __repr__(self):
        return f"Node({self.x}, {self.y}, {self.type})"

    def __hash__(self):
        return self.coordinates.__hash__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

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

    def update_relic_status(self, status: bool):
        if self._explored_for_relic and self._relic != status:
            raise ValueError(
                f"Can't change the relic status for {self}"
                ", the tile has already been explored"
            )

        self._relic = status
        self._explored_for_relic = True

    def update_reward_status(self, status: bool):
        if self._explored_for_reward and self._reward != status:
            raise ValueError(
                f"Can't change the reward status for {self}"
                ", the tile has already been explored"
            )

        self._reward = status
        self._explored_for_reward = True

    @property
    def is_unknown(self) -> bool:
        return self.type == NodeType.unknown

    @property
    def is_walkable(self) -> bool:
        return self.type != NodeType.asteroid

    @property
    def coordinates(self) -> tuple[int, int]:
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

        relic_nodes = set()
        for relic_id, (mask, xy) in enumerate(
            zip(obs["relic_nodes_mask"], obs["relic_nodes"])
        ):
            if mask:
                node = self.get_node(*xy)
                relic_nodes.add(node)
                self._relic_id_to_node[relic_id] = node

        explored_new_nodes = False
        for node in self:
            x, y = node.coordinates
            node.type = NodeType(int(tile_type[x][y]))
            if not node.is_unknown:
                node.energy = int(energy[x][y])
            else:
                node.energy = None

            if not node.explored_for_relic and not node.is_unknown:
                status = node in relic_nodes
                self._update_relic_status(x, y, status=status)
                explored_new_nodes = True

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if not node.is_unknown:
                self.get_opposite_node(*node.coordinates).type = node.type

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        Params.ALL_RELICS_FOUND = all_relics_found
        Params.ALL_REWARDS_FOUND = all_rewards_found

        if not Params.ALL_RELICS_FOUND:
            if self.num_relics_found == len(obs["relic_nodes_mask"]):
                # all relics found, mark all nodes as explored for relics
                Params.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)
                        explored_new_nodes = True

        if not Params.ALL_REWARDS_FOUND:
            if explored_new_nodes:
                self._update_reward_status_from_relics_distribution()

            if team_to_reward:
                self._update_reward_results(obs, team_to_reward)

            self._update_reward_status_from_reward_results()

    def _update_reward_status_from_relics_distribution(self):
        r = Params.RELIC_REWARD_RANGE
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

            if reward > len(nodes):
                print(
                    f"WARNING! Something wrong with reward result: {result}"
                    ", this result will be ignored.",
                    file=err,
                )
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

    def clear(self):
        for node in self:
            node.type = NodeType.unknown
            node.energy = None

    def update_nodes_by_expected_sensor_mask(self, expected_sensor_mask):
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                if expected_sensor_mask[y][x] == 1 and self.get_node(x, y).is_unknown:
                    self.get_node(x, y).type = NodeType.nebula

    def is_walkable(self, x, y):
        return self.get_node(x, y).is_walkable
