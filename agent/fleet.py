import numpy as np
from sys import stderr as err
from enum import IntEnum
from pathfinding import Grid, AStar, ResumableDijkstra

from .base import Params
from .space import Node, Space, NodeType


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


ACTION_TO_DIRECTION = {
    ActionType.center: (0, 0),
    ActionType.up: (0, -1),
    ActionType.right: (1, 0),
    ActionType.down: (0, 1),
    ActionType.left: (-1, 0),
    ActionType.sap: (0, 0),
}


def apply_action(x, y, action_type) -> tuple[int, int]:
    dx, dy = ACTION_TO_DIRECTION[action_type]
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
    last_position = path[0]
    for x, y in path[1:]:
        direction = ActionType.from_coordinates(last_position, (x, y))
        actions.append(Action(direction))
        last_position = (x, y)
    return actions


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0
        self.ships = [Ship(unit_id) for unit_id in range(Params.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def update(self, obs, space: Space):
        self.points = obs["team_points"][self.team_id]

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
            else:
                ship.clear()

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clear()

    def expected_sensor_mask(self):
        space_size = Params.SPACE_SIZE
        sensor_range = Params.UNIT_SENSOR_RANGE
        mask = np.zeros((space_size, space_size), dtype=np.int16)
        for ship in self:
            x, y = ship.coordinates
            for _y in range(
                max(0, y - sensor_range), min(space_size, y + sensor_range + 1)
            ):
                mask[_y][
                    max(0, x - sensor_range) : min(space_size, x + sensor_range + 1)
                ] = 1
        return mask


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task = None
        self.action_queue: list[Action] = []

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clear(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.action_queue = []

    def can_move(self) -> bool:
        return self.energy >= Params.UNIT_MOVE_COST

    def can_sap(self) -> bool:
        return self.energy >= Params.UNIT_SAP_COST

    def next_position(self) -> tuple[int, int]:
        if not self.can_move() or not self.action_queue:
            return self.coordinates
        return apply_action(*self.coordinates, action_type=self.action_queue[0].type)


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
