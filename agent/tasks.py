from .base import log, Global
from .exploration import VoidSeeker, RelicFinder
from .exploitation import VoidSinger
from .heal import Heal
from .sap import sap


def apply_tasks(state):

    for ship in state.fleet:
        task = ship.task
        if task is None:
            continue

        if task.completed(state, ship):
            ship.task = None
            continue

        if task.apply(state, ship):
            ship.task = task
        else:
            ship.task = None

    tasks = generate_tasks(state)

    scores = []
    for ship in state.fleet:
        if ship.task is None:
            for task in tasks:
                if task.ship and task.ship != ship:
                    continue
                score = task.evaluate(state, ship)
                if score <= 0:
                    continue
                scores.append({"ship": ship, "task": task, "score": score})

    scores = sorted(scores, key=lambda x: -x["score"])

    tasks_closed = set()
    ships_closed = set()

    for d in scores:
        ship = d["ship"]
        task = d["task"]

        if ship in ships_closed or task in tasks_closed:
            continue

        if task.apply(state, ship):
            ship.task = task

            ships_closed.add(ship)
            tasks_closed.add(task)

    sap(state)


def generate_tasks(state):
    return [
        *RelicFinder.generate_tasks(state),
        *VoidSeeker.generate_tasks(state),
        *VoidSinger.generate_tasks(state),
        *Heal.generate_tasks(state),
    ]
