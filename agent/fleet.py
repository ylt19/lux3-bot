from sys import stderr as err
import numpy as np
from enum import IntEnum
from pathfinding import Grid, AStar, ResumableDijkstra

from .base import SPACE_SIZE, MAX_ENERGY_PER_TILE
from .space import Node, Space


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
    def __init__(self, team_id, env_cfg):
        self.team_id: int = team_id
        self.points: int = 0

        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.unit_sap_range = env_cfg["unit_sap_range"]
        self.unit_sensor_range = env_cfg["unit_sensor_range"]

        self.ships = [Ship(unit_id) for unit_id in range(env_cfg["max_units"])]

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

    def create_actions_array(self):
        actions = np.zeros((len(self.ships), 3), dtype=int)
        for i, ship in enumerate(self.ships):
            if ship.action_queue:
                a = ship.action_queue[0]
                actions[i] = (a.type.value, a.dx, a.dy)
        return actions

    def show_tasks(self):
        print("Tasks:", file=err)
        for ship in self:
            print(f" - {ship} : {ship.task}", file=err)


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
        weights = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int16)
        for node in self._space:
            if not node.is_walkable:
                w = -1
            else:
                w = MAX_ENERGY_PER_TILE + 1 - node.energy
            weights[node.y][node.x] = w

        return Grid(weights)

    def get_resumable_search(self, start):
        return ResumableDijkstra(self.grid, start)
