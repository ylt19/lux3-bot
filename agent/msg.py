from .base import log, SPACE_SIZE, Task, Global, manhattan_distance, get_spawn_location
from .path import path_to_actions, Action, ActionType


DOWN = Action(ActionType.down)
UP = Action(ActionType.up)
LEFT = Action(ActionType.left)

# Happy New Year
HNY_MSG = {
    "size": (19, 5),
    "num_steps": 3,
    "min_sap_range": 4,
    "ship_tasks": [
        # H N Y
        {"position": (0, 0), "tasks": [(0, 4), (0, 4), (1, 2)]},
        {"position": (0, 2), "tasks": [(2, 0), (2, 2), None]},
        {"position": (2, 0), "tasks": [(0, 4), (0, 4), (-2, 4)]},
        # A E E
        {"position": (6, 0), "tasks": [(-2, 4), (-2, 0), (-2, 0)]},
        {"position": (6, 2), "tasks": [(-1, 0), (-2, 0), (-2, 0)]},
        {"position": (6, 4), "tasks": [(0, -4), (-2, 0), (-2, 0)]},
        # P W A
        {"position": (10, 0), "tasks": [(-2, 0), (-2, 4), (-2, 4)]},
        {"position": (10, 0), "tasks": [(0, 2), (0, 4), (0, 4)]},
        {"position": (10, 2), "tasks": [(-2, 0), None, (-1, 0)]},
        {"position": (8, 4), "tasks": [(0, -4), (0, -4), None]},
        # P - R
        {"position": (12, 0), "tasks": [(0, 4), (-2, 4), (0, 4)]},
        {"position": (12, 0), "tasks": [(2, 0), None, (2, 0)]},
        {"position": (14, 0), "tasks": [(0, 2), None, (-1, 2)]},
        {"position": (12, 2), "tasks": [(2, 0), None, (2, 2)]},
        # Y - -
        {"position": (16, 0), "tasks": [(1, 2), None, None]},
        {"position": (18, 0), "tasks": [(-2, 4), None, None]},
    ],
}

# Nice Play TY
NP_MSG = {
    "size": (15, 5),
    "num_steps": 3,
    "min_sap_range": 4,
    "ship_tasks": [
        # N P -
        {"position": (0, 0), "tasks": [(0, 4), (0, 4), None]},
        {"position": (0, 0), "tasks": [(2, 4), (2, 0), None]},
        {"position": (2, 0), "tasks": [(0, 4), (0, 2), None]},
        {"position": (0, 2), "tasks": [None, (2, 0), None]},
        # I L T
        {"position": (4, 0), "tasks": [(0, 4), (0, 4), (0, 4)]},
        # C L T
        {"position": (6, 0), "tasks": [(0, 4), None, (-4, 0)]},
        {"position": (8, 0), "tasks": [(-2, 0), None, (1, 2)]},
        {"position": (6, 4), "tasks": [(2, 0), (-2, 0), None]},
        # E A Y
        {"position": (10, 0), "tasks": [(0, 4), (-2, 4), (-2, 4)]},
        {"position": (12, 0), "tasks": [(-2, 0), (1, 2), None]},
        {"position": (10, 2), "tasks": [(2, 0), (-1, 0), None]},
        {"position": (10, 4), "tasks": [(2, 0), (0, -4), None]},
        # - Y -
        {"position": (14, 0), "tasks": [None, (-2, 4), None]},
    ],
}

# Nice Play TY small (if unit_sap_range == 3)
NP_MSG_SMALL = {
    "size": (11, 4),
    "num_steps": 3,
    "min_sap_range": 3,
    "ship_tasks": [
        # N P T
        {"position": (0, 0), "tasks": [(0, 3), (0, 3), None]},
        {"position": (0, 0), "tasks": [(2, 3), (2, 0), None]},
        {"position": (2, 0), "tasks": [(0, 3), (0, 2), (0, 3)]},
        {"position": (0, 2), "tasks": [None, (2, 0), None]},
        # I L T
        {"position": (3, 0), "tasks": [(0, 3), (0, 3), (-2, 0)]},
        # C L Y
        {"position": (4, 0), "tasks": [(0, 3), None, (1, 2)]},
        {"position": (6, 0), "tasks": [(-2, 0), None, (-2, 3)]},
        {"position": (4, 3), "tasks": [(2, 0), (-1, 0), None]},
        # E A -
        {"position": (7, 0), "tasks": [(0, 3), (-2, 3), None]},
        {"position": (7, 0), "tasks": [(2, 0), (0, 3), None]},
        {"position": (7, 2), "tasks": [(2, 0), (-2, 0), None]},
        {"position": (7, 3), "tasks": [(2, 0), (0, 0), None]},
        # - Y -
        {"position": (8, 0), "tasks": [None, (1, 2), None]},
        {"position": (10, 0), "tasks": [None, (-2, 3), None]},
    ],
}

# GG
GG_MSG = {
    "size": (11, 6),
    "num_steps": 2,
    "min_sap_range": 3,
    "ship_tasks": [
        # G G
        {"position": (4, 0), "tasks": [(-3, 0), (-3, 0)]},
        {"position": (1, 0), "tasks": [(-1, 1), (-1, 1)]},
        {"position": (0, 1), "tasks": [(0, 3), (0, 3)]},
        {"position": (0, 4), "tasks": [(1, 1), (1, 1)]},
        {"position": (1, 5), "tasks": [(3, 0), (3, 0)]},
        {"position": (4, 5), "tasks": [(0, -2), (0, -2)]},
        {"position": (4, 3), "tasks": [(-2, 0), None]},
        {"position": (2, 3), "tasks": [None, (2, 0)]},
        # G G
        {"position": (10, 0), "tasks": [(-3, 0), (-3, 0)]},
        {"position": (7, 0), "tasks": [(-1, 1), (-1, 1)]},
        {"position": (6, 1), "tasks": [(0, 3), (0, 3)]},
        {"position": (6, 4), "tasks": [(1, 1), (1, 1)]},
        {"position": (7, 5), "tasks": [(3, 0), (3, 0)]},
        {"position": (10, 5), "tasks": [(0, -2), (0, -2)]},
        {"position": (10, 3), "tasks": [(-2, 0), None]},
        {"position": (8, 3), "tasks": [None, (2, 0)]},
    ],
}

# Have a good day
GD_MSG = {
    "size": (12, 4),
    "num_steps": 7,
    "min_sap_range": 3,
    "check_occupancies": True,
    "ship_tasks": [
        # H A G D
        {
            "position": (2, 0),
            "tasks": [(0, 3), None, None, None, (-2, 0), None, (-1, 0)],
        },
        {
            "position": (0, 3),
            "tasks": [(0, -3), None, (2, -3), None, (0, -3), None, (0, -3)],
        },
        {
            "position": (2, 3),
            "tasks": [None, None, (0, -3), None, (-2, 0), None, (-1, 0)],
        },
        {
            "position": (2, 2),
            "tasks": [(-2, 0), None, (-1, 0), None, (0, 1), DOWN, (0, -3)],
        },
        # A - O A
        {
            "position": (5, 0),
            "tasks": [(0, 3), None, None, None, (0, 3), None, None],
        },
        {
            "position": (5, 0),
            "tasks": [(-2, 3), None, None, None, (-2, 0), None, (0, 3)],
        },
        {
            "position": (3, 3),
            "tasks": [None, None, None, None, (0, -3), None, (2, -3)],
        },
        {
            "position": (5, 2),
            "tasks": [(-1, 0), DOWN, None, None, (-2, 0), UP, (-1, 0)],
        },
        # V - O Y
        {
            "position": (6, 0),
            "tasks": [(1, 3), None, None, None, (0, 3), None, (1, 2)],
        },
        {
            "position": (8, 0),
            "tasks": [(-1, 3), None, None, None, (-2, 0), None, None],
        },
        {
            "position": (8, 0),
            "tasks": [None, None, None, None, (0, 3), None, (-2, 3)],
        },
        {
            "position": (7, 3),
            "tasks": [None, LEFT, None, None, (2, 0), None, None],
        },
        # E - D -
        {
            "position": (9, 0),
            "tasks": [(0, 3), None, None, None, (0, 3), None, None],
        },
        {
            "position": (11, 0),
            "tasks": [(-2, 0), None, None, None, (-1, 0), None, (0, 2)],
        },
        {
            "position": (11, 2),
            "tasks": [(-2, 0), DOWN, None, None, (-1, 0), None, None],
        },
        {
            "position": (11, 3),
            "tasks": [(-2, 0), None, None, None, (0, -3), None, None],
        },
    ],
}


class MsgTask(Task):

    def __init__(self, target, path, tasks):
        super().__init__(target)
        self.path = path
        self.tasks = tasks

    def __repr__(self):
        return f"{self.__class__.__name__}{self.target.coordinates}"

    def completed(self, state, ship):
        return False

    def apply(self, state, ship):
        return True


def print_msg(state):

    if Global.NUM_WINS == 0:
        msg = GG_MSG
    elif Global.NUM_WINS == state.match_number:
        msg = GD_MSG
    else:
        if Global.UNIT_SAP_RANGE == 3:
            msg = NP_MSG_SMALL
        elif Global.UNIT_SAP_RANGE >= 4:
            msg = NP_MSG
        else:
            return

    p = Global.Params

    if not p.MSG_TASK or p.MSG_TASK_FINISHED:
        return

    if p.MSG_TASK_STARTED:
        continue_to_print(state, msg)
        return

    if Global.UNIT_SAP_RANGE < msg["min_sap_range"]:
        return

    if apply_tasks(state, msg):
        p.MSG_TASK_STARTED = True


def continue_to_print(state, msg):
    num_ships = sum(isinstance(x.task, MsgTask) for x in state.fleet)
    if num_ships != len(msg["ship_tasks"]):
        # We've lost some ships.
        Global.Params.MSG_TASK_STARTED = False
        Global.Params.MSG_TASK_FINISHED = False
        for ship in state.fleet:
            if isinstance(ship.task, MsgTask):
                ship.task = None
        return

    num_msg_ships = 0
    for ship in state.fleet:
        task = ship.task

        if not isinstance(task, MsgTask):
            continue

        if len(task.path) > 1:
            if not ship.can_move():
                log(
                    f"Task {task} cannot be completed, {ship} cannot move, step={state.global_step}"
                )
                ship.task = None
                continue

            rs = state.grid.resumable_search(ship.unit_id)
            new_path = rs.find_path(
                node=task.target.coordinates, time=len(task.path) - 2
            )

            if not new_path:
                ship.task = None
                ship.action_queue = []
                continue

            task.path = new_path
            ship.action_queue = path_to_actions(new_path)
            num_msg_ships += 1

        else:
            if not task.tasks:
                ship.task = None
                continue

            next_action = task.tasks[0]
            if isinstance(next_action, (tuple, list)):
                ship.action_queue = [Action(ActionType.sap, *next_action)]
            elif isinstance(next_action, Action):
                ship.action_queue = [next_action]
            else:
                ship.action_queue = []

            task.tasks = task.tasks[1:]
            num_msg_ships += 1

    if num_msg_ships == 0:
        Global.Params.MSG_TASK_STARTED = False
        Global.Params.MSG_TASK_FINISHED = True


def apply_tasks(state, msg):
    if state.steps_left_in_match() < 20:
        return False

    ships = []
    for ship in state.fleet:
        if ship.energy > 100:
            ships.append(ship)

    if len(ships) < len(msg["ship_tasks"]):
        return False

    spawn_position = get_spawn_location(state.team_id)

    top_left_options = []
    for top_left_node in state.space:
        top_left = top_left_node.coordinates
        if top_left[0] == 0 or top_left[1] == 0:
            continue

        bottom_right = (
            top_left[0] + msg["size"][0] - 1,
            top_left[1] + msg["size"][1] - 1,
        )
        if bottom_right[0] >= SPACE_SIZE - 1 or bottom_right[1] >= SPACE_SIZE - 1:
            continue

        if state.team_id == 0:
            distance = manhattan_distance(spawn_position, top_left)
        else:
            distance = manhattan_distance(spawn_position, bottom_right)

        if distance >= SPACE_SIZE / 2 - 1:
            continue

        top_left_options.append(top_left)

    top_left_options = sorted(
        top_left_options, key=lambda x: manhattan_distance(x, spawn_position)
    )

    for top_left in top_left_options:
        ship_to_task = apply_to_position(state, ships, top_left, msg)
        if ship_to_task:
            for ship, task in ship_to_task.items():
                goal = state.space.get_node(*task["goal"])
                path = task["path"]

                msg_task = MsgTask(goal, path, task["tasks"])
                ship.task = msg_task
                ship.action_queue = path_to_actions(path)

                log(
                    f"create msg task {msg_task}, path={msg_task.path}, tasks={msg_task.tasks}"
                )
            return True

    return False


def apply_to_position(state, ships, top_left, msg):
    top_left_x, top_left_y = top_left
    ship_tasks = msg["ship_tasks"]
    num_steps = msg["num_steps"]

    def get_absolute_position(position):
        return top_left_x + position[0], top_left_y + position[1]

    # check for reachability
    ship_to_task = {}
    for task in ship_tasks:
        goal = get_absolute_position(task["position"])
        num_saps = sum(
            isinstance(x, tuple) or (isinstance(x, Action) and x.type == ActionType.sap)
            for x in task["tasks"]
        )
        num_moves = sum(
            isinstance(x, Action) and x.type not in (ActionType.sap, ActionType.center)
            for x in task["tasks"]
        )
        task_energy = (
            Global.UNIT_SAP_COST * num_saps + Global.UNIT_MOVE_COST * num_moves
        )

        best_ship, min_distance = None, float("inf")
        for ship in ships:
            if ship in ship_to_task:
                continue

            if (
                ship.energy
                < task_energy + Global.NEBULA_ENERGY_REDUCTION * num_steps + 5
            ):
                continue

            rs = state.grid.resumable_search(ship.unit_id)
            path = rs.find_path(goal)
            distance = len(path)
            if state.match_step + distance + num_steps < Global.MAX_STEPS_IN_MATCH - 1:
                if distance < min_distance:
                    best_ship, min_distance = ship, distance

        if not best_ship:
            return

        rs = state.grid.resumable_search(best_ship.unit_id)
        path = rs.find_path(goal)
        ship_to_task[best_ship] = {"tasks": task["tasks"], "goal": goal, "path": path}

    # find the path with exact length
    # thus, all ships will reach the targets in the same time
    max_length = max(len(x["path"]) for x in ship_to_task.values())
    for ship, task in ship_to_task.items():
        rs = state.grid.resumable_search(ship.unit_id)
        path = rs.find_path(task["goal"], time=max_length)
        if not path:
            return

        if msg.get("check_occupancies"):
            position, time = task["goal"], max_length
            for t in task["tasks"]:
                time += 1
                if isinstance(t, Action):
                    dx, dy = t.type.to_direction()
                    position = position[0] + dx, position[1] + dy
                    if state.grid.reservation_table.is_reserved(time, position):
                        return

        ship_to_task[ship]["path"] = path

    return ship_to_task
