from sys import stderr as err
from collections import defaultdict

from .base import manhattan_distance
from .path import (
    path_to_actions,
    find_closest_target,
    estimate_energy_cost,
    find_path_in_dynamic_environment,
)
from .tasks import HarvestTask, GatherEnergy


def harvest(state):
    space = state.space
    fleet = state.fleet

    booked_nodes = set()
    for ship in fleet:
        if isinstance(ship.task, HarvestTask):
            target = ship.task.coordinates
            if set_path_to_target(state, ship, target):
                booked_nodes.add(state.space.get_node(*target))
            else:
                ship.task = None

    targets = set()
    for n in space.reward_nodes:
        if n.is_walkable and n not in booked_nodes:
            targets.add(n.coordinates)
    if not targets:
        return

    steps_left = state.steps_left_in_match()
    for ship in fleet:
        if ship.task and not isinstance(ship.task, GatherEnergy):
            continue

        ship_targets = [
            x for x in targets if manhattan_distance(ship.coordinates, x) < steps_left
        ]

        target, _ = find_closest_target(state, ship.coordinates, ship_targets)
        if not target:
            ship.task = None
            continue

        if set_path_to_target(state, ship, target):
            targets.remove(target)
        else:
            ship.task = None


def set_path_to_target(state, ship, target) -> bool:
    if ship.coordinates == target:
        return True

    if not ship.can_move():
        return False

    path = find_path_in_dynamic_environment(
        state, start=ship.coordinates, goal=target, ship_energy=ship.energy
    )
    if not path:
        return False

    energy = estimate_energy_cost(state.space, path)

    if ship.energy < energy:
        return False

    ship.task = HarvestTask(state.space.get_node(*target))
    ship.action_queue = path_to_actions(path)
    return True
