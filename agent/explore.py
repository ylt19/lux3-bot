from sys import stderr as err

from .base import Step, is_team_sector, nearby_positions
from .space import Space
from .fleet import Fleet, PathFinder, path_to_actions
from .tasks import FindRelicNodes, FindRewardNodes


def explore(step: Step, space: Space, fleet: Fleet):
    if step.global_step >= 150:
        delete_tasks(fleet, (FindRelicNodes, FindRewardNodes))
        return

    find_relics(space, fleet)
    find_rewards(space, fleet)


def find_relics(space: Space, fleet: Fleet):
    if space.all_relics_found:
        delete_tasks(fleet, FindRelicNodes)
        return

    nodes_to_explore = set()
    for node in space:
        if not node.explored_for_relic and is_team_sector(
            fleet.team_id, *node.coordinates
        ):
            nodes_to_explore.add(node)

    finder = PathFinder(space)

    for ship in fleet:
        if ship.task and not isinstance(ship.task, FindRelicNodes):
            continue

        if not nodes_to_explore:
            ship.task = None
            ship.action_queue = []
            continue

        rs = finder.get_resumable_search(start=ship.coordinates)

        min_distance = 0
        target = None
        for node in nodes_to_explore:
            distance = rs.distance(node.coordinates)
            if target is None or distance < min_distance:
                min_distance = distance
                target = node

        if target is None or min_distance == float("inf"):
            ship.task = None
            ship.action_queue = []
            continue

        path = finder.find_path(ship.coordinates, target.coordinates)
        ship.task = FindRelicNodes()
        ship.action_queue = path_to_actions(path)

        for x, y in path:
            for _x, _y in nearby_positions(x, y, fleet.unit_sensor_range):
                node = space.get_node(_x, _y)
                if node in nodes_to_explore:
                    nodes_to_explore.remove(node)


def find_rewards(space: Space, fleet: Fleet):
    if space.all_rewards_found:
        delete_tasks(fleet, FindRewardNodes)
        return

    finder = PathFinder(space)

    booked_nodes = set()
    for ship in fleet:
        if not isinstance(ship.task, FindRewardNodes):
            continue

        target = ship.task.node
        if target.explored_for_reward:
            ship.task = None
            ship.action_queue = []
            continue

        path = finder.find_path(ship.coordinates, target.coordinates)
        if path and path[-1] == target.coordinates:
            booked_nodes.add(target)
            ship.action_queue = path_to_actions(path)
        else:
            ship.task = None
            ship.action_queue = []

    target_nodes = set()
    for node in space:
        if (
            not node.explored_for_reward
            and is_team_sector(fleet.team_id, *node.coordinates)
            and node not in booked_nodes
        ):
            target_nodes.add(node)

    for ship in fleet:
        if ship.task or not target_nodes:
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
            ship.task = None
            ship.action_queue = []
            continue

        target_nodes.remove(target)
        path = finder.find_path(ship.coordinates, target.coordinates)
        ship.task = FindRewardNodes(target)
        ship.action_queue = path_to_actions(path)


def delete_tasks(fleet, task_type):
    for ship in fleet:
        if isinstance(ship.task, task_type):
            ship.task = None
            ship.action_queue = []
