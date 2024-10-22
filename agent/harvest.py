from sys import stderr as err

from .path import PathFinder, path_to_actions, estimate_energy_cost
from .tasks import HarvestTask, GatherEnergy


def harvest(state):
    space = state.space
    fleet = state.fleet

    finder = PathFinder(state.explored_space)

    booked_nodes = set()
    for ship in fleet:
        if isinstance(ship.task, HarvestTask):
            target = ship.task.coordinates
            if set_path_to_target(state, ship, target, finder):
                booked_nodes.add(state.space.get_node(*target))
            else:
                ship.task = None

    targets = set()
    for n in space.reward_nodes:
        if n.is_walkable and n not in booked_nodes:
            targets.add(n.coordinates)
    if not targets:
        return

    for ship in fleet:
        if ship.task and not isinstance(ship.task, GatherEnergy):
            continue

        target, _ = finder.find_closest_target(ship.coordinates, targets)
        if not target:
            continue

        if set_path_to_target(state, ship, target, finder):
            targets.remove(target)


def set_path_to_target(state, ship, target, finder) -> bool:
    if ship.coordinates == target:
        return True

    if not ship.can_move():
        return False

    path = finder.find_path(ship.coordinates, target)
    energy = estimate_energy_cost(state.explored_space, path)

    if ship.energy < energy:
        return False

    ship.task = HarvestTask(state.space.get_node(*target))
    ship.action_queue = path_to_actions(path)
    return True
