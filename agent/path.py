import copy
from sys import stderr as err
from enum import IntEnum
from pathfinding import Grid, AStar, SpaceTimeAStar, ResumableDijkstra, ReservationTable

from .base import Params, is_inside, nearby_positions, manhattan_distance
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


def actions_to_path(start, actions):
    p = start
    path = [p]
    for a in actions:
        p = apply_action(*p, action=a.type)
        path.append(p)
    return path


def estimate_energy_cost(space: Space, path: list[tuple[int, int]]):
    if len(path) <= 1:
        return 0

    energy = 0
    last_position = path[0]
    for x, y in path[1:]:
        node = space.get_node(x, y)
        if node.energy is not None:
            energy -= node.energy
        else:
            energy -= Params.HIDDEN_NODE_ENERGY

        if node.type == NodeType.nebula:
            energy += Params.NEBULA_ENERGY_REDUCTION

        if (x, y) != last_position:
            energy += Params.UNIT_MOVE_COST

    return energy


def allowed_movements(x, y, space):
    actions = []
    for action in (ActionType.right, ActionType.left, ActionType.up, ActionType.down):
        _x, _y = apply_action(x, y, action)
        if is_inside(_x, _y) and space.is_walkable(_x, _y):
            actions.append(action)
    return actions


def find_path_in_dynamic_environment(state, start, goal, ship_energy=None):
    grid = state.energy_grid
    reservation_table = copy.copy(state.reservation_table)

    if ship_energy is not None:
        _add_opp_ships(reservation_table, state, ship_energy)

    finder = SpaceTimeAStar(grid)
    path = finder.find_path_with_length_limit(
        start,
        goal,
        max_length=state.steps_left_in_match(),
        reservation_table=reservation_table,
    )
    return path


def _add_opp_ships(rt, state, ship_energy):
    for opp_ship in state.opp_fleet:
        if opp_ship.energy < ship_energy:
            continue

        opp_coord = opp_ship.coordinates
        for p in nearby_positions(*opp_coord, distance=2):
            if manhattan_distance(p, opp_coord) <= 2:
                rt.add_vertex_constraint(p, time=1)


def find_closest_target(state, start, targets):
    if not targets:
        return None, float("inf")

    rs = ResumableDijkstra(state.obstacle_grid, start)

    target, min_distance = None, float("inf")
    for t in targets:
        d = rs.distance(t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance
