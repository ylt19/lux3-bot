import numpy as np
from .base import log, Task, Global, manhattan_distance
from .path import path_to_actions, Action, ActionType

from pathfinding import Grid, SpaceTimeAStar, ReservationTable

# Happy New Year Message
HNY_MSG = {
    "top_left_options": {
        0: [
            *[(1, y) for y in range(1, 14, 2)],
            *[(2, y) for y in range(1, 14, 2)],
            *[(3, y) for y in range(1, 14, 2)],
        ],
        1: [
            *[(4, y) for y in range(18, 5, -2)],
            *[(3, y) for y in range(18, 5, -2)],
            *[(2, y) for y in range(18, 5, -2)],
        ],
    },
    "size": (19, 5),
    "num_steps": 3,
    "min_sap_range": 4,
    "ship_tasks": [
        # H N Y
        {"position": (0, 0), "sap": [(0, 4), (0, 4), (1, 2), None]},
        {"position": (0, 2), "sap": [(2, 0), (2, 2), None, None]},
        {"position": (2, 0), "sap": [(0, 4), (0, 4), (-2, 4), None]},
        # A E E
        {"position": (6, 0), "sap": [(-2, 4), (-2, 0), (-2, 0), None]},
        {"position": (6, 2), "sap": [(-1, 0), (-2, 0), (-2, 0), None]},
        {"position": (6, 4), "sap": [(0, -4), (-2, 0), (-2, 0), None]},
        # P W A
        {"position": (10, 0), "sap": [(-2, 0), (-2, 4), (-2, 4), None]},
        {"position": (10, 0), "sap": [(0, 2), (0, 4), (0, 4), None]},
        {"position": (10, 2), "sap": [(-2, 0), None, (-1, 0), None]},
        {"position": (8, 4), "sap": [(0, -4), (0, -4), None, None]},
        # P - R
        {"position": (12, 0), "sap": [(0, 4), (-2, 4), (0, 4), None]},
        {"position": (12, 0), "sap": [(2, 0), None, (2, 0), None]},
        {"position": (14, 0), "sap": [(0, 2), None, (-1, 2), None]},
        {"position": (12, 2), "sap": [(2, 0), None, (2, 2), None]},
        # Y - -
        {"position": (16, 0), "sap": [(1, 2), None, None, None]},
        {"position": (18, 0), "sap": [(-2, 4), None, None, None]},
    ],
}


class MsgTask(Task):

    def __init__(self, target, path, sap, num_ships):
        super().__init__(target)
        self.path = path
        self.sap = sap
        self.num_ships = num_ships

    def __repr__(self):
        return f"{self.__class__.__name__}{self.target.coordinates}"

    def completed(self, state, ship):
        num_ships = sum(isinstance(x.task, MsgTask) for x in state.fleet)
        if num_ships != self.num_ships:
            # We've lost some ships.
            return True

        return False

    def apply(self, state, ship):
        if len(self.path) > 1:
            if not ship.can_move():
                log(
                    f"Task {self} cannot be completed, {ship} cannot move, step={state.global_step}"
                )
                return False

            self.path = self.path[1:]
            ship.action_queue = path_to_actions(self.path)

        else:
            if not self.sap:
                return False

            if self.sap[0]:
                ship.action_queue = [Action(ActionType.sap, *self.sap[0])]

            self.sap = self.sap[1:]

        return True


def print_msg(state):
    msg = HNY_MSG

    p = Global.Params
    if (
        not p.MSG_TASK
        or p.MSG_GENERATED
        or Global.UNIT_SAP_RANGE < msg["min_sap_range"]
    ):
        return

    if apply_tasks(state, msg):
        p.MSG_GENERATED = True
    else:
        return


def apply_tasks(state, msg):
    if state.steps_left_in_match() < 20:
        return False

    ships = []
    for ship in state.fleet:
        if ship.energy > 100:
            ships.append(ship)

    if len(ships) < 16:
        return False

    for top_left in msg["top_left_options"][state.team_id]:
        ship_to_task = apply_to_position(state, ships, top_left, msg)
        if ship_to_task:
            for ship, task in ship_to_task.items():
                goal = state.space.get_node(*task["goal"])
                path = task["path"]

                msg_task = MsgTask(goal, path, task["sap"], len(ship_to_task))
                ship.task = msg_task
                ship.action_queue = path_to_actions(path)

                log(
                    f"create msg task {msg_task}, path={msg_task.path}, sap={msg_task.sap}"
                )
            return True

    return False


def apply_to_position(state, ships, top_left, msg):
    top_left_x, top_left_y = top_left
    ship_tasks = msg["ship_tasks"]
    num_msg_steps = msg["num_steps"]

    def get_absolute_position(position):
        return top_left_x + position[0], top_left_y + position[1]

    # check for reachability
    ship_to_task = {}
    for task in ship_tasks:
        goal = get_absolute_position(task["position"])
        num_saps = sum(x is not None for x in task["sap"])

        best_ship, min_distance = None, float("inf")
        for ship in ships:
            if ship in ship_to_task:
                continue

            if (
                ship.energy
                < Global.UNIT_SAP_COST * num_saps
                + Global.NEBULA_ENERGY_REDUCTION * num_msg_steps
                + 5
            ):
                continue

            distance = manhattan_distance(ship.coordinates, goal)
            if (
                state.match_step + distance + num_msg_steps
                < Global.MAX_STEPS_IN_MATCH - 1
            ):
                if distance < min_distance:
                    best_ship, min_distance = ship, distance

        if not best_ship:
            return

        ship_to_task[best_ship] = {"sap": task["sap"], "goal": goal, "path": []}

    # find path
    finder, reservation_table = create_finder(state, energy_ground=10)
    for ship, task in ship_to_task.items():
        path = finder.find_path_with_length_limit(
            ship.coordinates,
            task["goal"],
            max_length=state.steps_left_in_match() - num_msg_steps,
            reservation_table=reservation_table,
        )
        if not path:
            return
        ship_to_task[ship]["path"] = path

    # find the best path
    finder, reservation_table = create_finder(
        state, energy_ground=Global.UNIT_MOVE_COST
    )
    max_length = max(len(x["path"]) for x in ship_to_task.values())
    for ship, task in ship_to_task.items():
        path = finder.find_path_with_exact_length(
            ship.coordinates,
            task["goal"],
            length=max_length,
            reservation_table=reservation_table,
        )
        if not path:
            return
        ship_to_task[ship]["path"] = path

    return ship_to_task


def create_finder(state, energy_ground):
    from .state import add_dynamic_environment

    def energy_to_weight(energy):
        if energy < energy_ground:
            return energy_ground - energy + 1
        return Global.Params.ENERGY_TO_WEIGHT_BASE ** (energy_ground - energy)

    weights = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
    for node in state.space:
        weights[node.y][node.x] = energy_to_weight(node.energy_gain)

    grid = Grid(weights, pause_action_cost="node.weight")

    reservation_table = ReservationTable(grid)
    add_dynamic_environment(reservation_table, state)

    finder = SpaceTimeAStar(grid)
    return finder, reservation_table
