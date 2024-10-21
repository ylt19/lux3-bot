import numpy as np
from sys import stderr as err
from enum import IntEnum
from pathfinding import Grid, AStar, ResumableDijkstra

from .base import Params, is_inside
from .space import Space, NodeType

DIRECTIONS = [
    (0, 0),  # center
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  #  down
    (-1, 0),  # left
    (0, 0),  # sap
]


class ActionType(IntEnum):
    center = 0
    up = 1
    right = 2
    down = 3
    left = 4
    sap = 5

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @classmethod
    def from_coordinates(cls, current_position, next_position):
        dx = next_position[0] - current_position[0]
        dy = next_position[1] - current_position[1]

        if dx < 0:
            return ActionType.left
        elif dx > 0:
            return ActionType.right
        elif dy < 0:
            return ActionType.up
        elif dy > 0:
            return ActionType.down
        else:
            return ActionType.center

    def to_direction(self):
        return DIRECTIONS[self]


def apply_action(x: int, y: int, action: ActionType) -> tuple[int, int]:
    dx, dy = action.to_direction()
    return x + dx, y + dy


class Action:
    def __init__(self, action_type: ActionType, dx: int = 0, dy: int = 0):
        self.type = action_type
        self.dx = dx
        self.dy = dy

    def __repr__(self):
        if self.type == ActionType.sap:
            return f"sup({self.dx}, {self.dy})"
        return str(self.type)


def path_to_actions(path):
    actions = []
    if not path:
        return actions

    last_position = path[0]
    for x, y in path[1:]:
        direction = ActionType.from_coordinates(last_position, (x, y))
        actions.append(Action(direction))
        last_position = (x, y)

    return actions


def estimate_energy_cost(space: Space, path: list[tuple[int, int]]):
    if len(path) <= 1:
        return 0

    energy = 0
    for x, y in path[1:]:
        node = space.get_node(x, y)
        if node.energy is not None:
            energy -= node.energy
        else:
            energy -= Params.HIDDEN_NODE_ENERGY

        if node.type == NodeType.nebula:
            energy += Params.NEBULA_ENERGY_REDUCTION

    return energy


def allowed_movements(x, y, space):
    actions = []
    for action in (ActionType.right, ActionType.left, ActionType.up, ActionType.down):
        _x, _y = apply_action(x, y, action)
        if is_inside(_x, _y) and space.is_walkable(_x, _y):
            actions.append(action)
    return actions


class PathFinder:
    def __init__(self, space, algorithm=AStar):
        self._grid = None
        self._finder = None
        self._space = space
        self._algorithm = algorithm

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self._create_grid()
        return self._grid

    @property
    def finder(self):
        if self._finder is None:
            self._finder = self._algorithm(self.grid)
        return self._finder

    def find_path(self, start, goal):
        return self.finder.find_path(start, goal)

    def cost(self, path):
        return self.grid.calculate_cost(path)

    def _create_grid(self):
        weights = np.zeros((Params.SPACE_SIZE, Params.SPACE_SIZE), np.int16)
        for node in self._space:

            if not node.is_walkable:
                w = -1
            else:
                node_energy = node.energy
                if node_energy is None:
                    node_energy = Params.HIDDEN_NODE_ENERGY

                w = Params.MAX_ENERGY_PER_TILE + 1 - node_energy

            if node.type == NodeType.nebula:
                w += Params.NEBULA_ENERGY_REDUCTION

            weights[node.y][node.x] = w

        return Grid(weights)

    def get_resumable_search(self, start):
        return ResumableDijkstra(self.grid, start)

    def find_closest_target(self, start, targets, rs=None):
        if not targets:
            return None, float("inf")

        if not rs:
            rs = self.get_resumable_search(start)

        target, min_distance = None, float("inf")
        for t in targets:
            d = rs.distance(t)
            if d < min_distance:
                target, min_distance = t, d

        return target, min_distance
