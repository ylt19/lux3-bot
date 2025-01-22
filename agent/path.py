import copy
from enum import IntEnum
from pathfinding import (
    Grid,
    AStar,
    SpaceTimeAStar,
    ResumableBFS,
    ResumableDijkstra,
    ReservationTable,
)

from .base import (
    Global,
    is_inside,
    nearby_positions,
    manhattan_distance,
    cardinal_positions,
)
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

    def transpose(self, reflective=False):
        if reflective:
            if self == ActionType.up:
                return ActionType.right
            elif self == ActionType.right:
                return ActionType.up
            elif self == ActionType.left:
                return ActionType.down
            elif self == ActionType.down:
                return ActionType.left
            else:
                return self
        else:
            if self == ActionType.up:
                return ActionType.left
            elif self == ActionType.left:
                return ActionType.up
            elif self == ActionType.right:
                return ActionType.down
            elif self == ActionType.down:
                return ActionType.right
            else:
                return self


def apply_action(x: int, y: int, action: ActionType) -> tuple[int, int]:
    dx, dy = action.to_direction()
    return x + dx, y + dy


class Action:
    def __init__(self, action_type: ActionType, dx: int = 0, dy: int = 0):
        self.type = action_type
        self.dx = int(dx)
        self.dy = int(dy)

    def __repr__(self):
        if self.type == ActionType.sap:
            return f"sup({self.dx}, {self.dy})"
        return str(self.type)


def path_to_actions(path):

    if len(path) == 0:
        return []

    if len(path) == 1:
        return [Action(ActionType.center)]

    actions = []
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
            energy -= Global.HIDDEN_NODE_ENERGY

        if node.type == NodeType.nebula:
            energy += Global.NEBULA_ENERGY_REDUCTION

        if (x, y) != last_position:
            energy += Global.UNIT_MOVE_COST

    return energy


def allowed_movements(x, y, space):
    actions = []
    for action in (ActionType.right, ActionType.left, ActionType.up, ActionType.down):
        _x, _y = apply_action(x, y, action)
        if is_inside(_x, _y) and space.is_walkable(_x, _y):
            actions.append(action)
    return actions


def find_path_in_dynamic_environment(state, start, goal, ship_energy=None, grid=None):
    grid = grid or state.grid.energy
    reservation_table = copy.copy(state.grid.reservation_table)

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
        for p in cardinal_positions(*opp_coord):
            rt.add_vertex_constraint(time=1, node=p)


def get_reachable_nodes(state, start):
    grid = state.obstacle_grid
    if not state.space.is_walkable(*start):
        # There is an asteroid at our starting position.
        # However, we can still move to an adjacent free tile.
        # We need to clear the obstacle from the grid,
        # as our pathfinding cannot handle obstacles at the start.
        grid = copy.copy(grid)
        grid.remove_obstacle(start)

    rs = ResumableBFS(grid, start)
    steps_left = state.steps_left_in_match()

    reachable_nodes = []
    for node in state.space:
        d = rs.distance(node.coordinates)
        if d < steps_left:
            reachable_nodes.append(node)

    return reachable_nodes


def find_closest_target(state, start, targets):
    if not targets:
        return None, float("inf")

    grid = copy.copy(state.obstacle_grid)

    for opp_ship in state.opp_fleet:

        if opp_ship.energy > 0:
            x, y = opp_ship.coordinates
            w = grid.get_weight((x, y))
            if w >= 0:
                grid.update_weight(
                    (x, y), w + opp_ship.energy * Global.UNIT_ENERGY_VOID_FACTOR
                )

            for x_, y_ in cardinal_positions(x, y):
                w = grid.get_weight((x_, y_))
                if w >= 0:
                    grid.update_weight(
                        (x_, y_), w + opp_ship.energy * Global.UNIT_ENERGY_VOID_FACTOR
                    )

    # If there is an asteroid at our starting position, we can still move to an adjacent free tile.
    # We need to clear the obstacle from the grid, as our pathfinding cannot handle obstacles at the start.
    grid.remove_obstacle(start)

    rs = ResumableDijkstra(grid, start)

    target, min_distance = None, float("inf")
    for t in targets:
        d = rs.distance(t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance
