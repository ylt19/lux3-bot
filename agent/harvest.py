from sys import stderr as err

from .path import PathFinder, path_to_actions, estimate_energy_cost
from .tasks import HarvestTask


def harvest(state):
    space = state.space
    fleet = state.fleet

    finder = PathFinder(state.explored_space)

    booked_nodes = set()
    for ship in fleet:
        if not isinstance(ship.task, HarvestTask):
            continue

        target = space.get_node(*ship.task.coordinates)
        if ship.node == target:
            booked_nodes.add(target)
            ship.action_queue = []
            continue

        path = finder.find_path(ship.coordinates, target.coordinates)
        if path and path[-1] == target.coordinates:
            booked_nodes.add(target)
            ship.action_queue = path_to_actions(path)
        else:
            ship.task = None

    target_nodes = set()
    for n in space.reward_nodes:
        if n.is_walkable and n not in booked_nodes:
            target_nodes.add(n)
    if not target_nodes:
        return

    for ship in fleet:
        if ship.task:
            continue

        rs = finder.get_resumable_search(start=ship.coordinates)

        min_distance = 0
        target = None
        for node in target_nodes:
            distance = rs.distance(node.coordinates)
            if target is None or distance < min_distance:
                min_distance = distance
                target = node

        if target is None or min_distance == float("inf"):
            continue

        target_nodes.remove(target)
        path = finder.find_path(ship.coordinates, target.coordinates)
        energy = estimate_energy_cost(state.explored_space, path)

        if ship.energy >= energy:
            ship.task = HarvestTask(target)
            ship.action_queue = path_to_actions(path)
        else:
            ship.task = None
