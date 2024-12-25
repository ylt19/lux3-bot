from .base import log, Global
from .exploration import VoidSeeker, RelicFinder
from .exploitation import VoidSinger
from .heal import heal
from .sap import sap


def apply_tasks(state):

    for ship in state.fleet:
        if ship.task is not None and ship.task.completed(state, ship):
            ship.task = None

    tasks = generate_tasks(state)

    scores = []
    for ship in state.fleet:
        if ship.task is None:
            for task in tasks:
                score = task.evaluate(state, ship)
                if score <= 0:
                    continue
                scores.append({"ship": ship, "task": task, "score": score})

    scores = sorted(scores, key=lambda x: -x["score"])

    picked_tasks = set()

    for d in scores:
        ship = d["ship"]
        task = d["task"]
        if ship.task is not None or task in picked_tasks:
            continue

        ship.task = task
        picked_tasks.add(task)

    for ship in state.fleet:
        if ship.task is not None:
            ship.task.apply(state, ship)

    for ship in state.fleet:
        if ship.task is None:
            heal(state, ship)

    sap(state)


def generate_tasks(state):
    return [
        *RelicFinder.generate_tasks(state),
        *VoidSeeker.generate_tasks(state),
        *VoidSinger.generate_tasks(state),
    ]
