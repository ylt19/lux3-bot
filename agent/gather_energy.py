from .base import Params
from .path import PathFinder, path_to_actions, estimate_energy_cost
from .space import NodeType
from .state import State
from .tasks import GatherEnergy


def gather_energy(state: State):

    max_energy = 0
    for node in state.space:
        if node.energy is not None:
            max_energy = max(max_energy, node.energy)

    targets = []
    for node in state.space:
        if node.energy is not None and node.energy >= max_energy - 1:
            targets.append(node.coordinates)

    if not targets:
        return

    finder = PathFinder(state)

    for ship in state.fleet:
        if ship.task and not isinstance(ship.task, GatherEnergy):
            continue

        available_locations = finder.get_available_locations(ship.coordinates)
        targets = get_positions_with_max_energy(state, available_locations)

        target, _ = finder.find_closest_target(ship.coordinates, targets)
        if not target:
            ship.task = None
            continue

        path = finder.find_path(ship.coordinates, target, dynamic=True)
        energy = estimate_energy_cost(state.space, path)

        if ship.energy >= energy:
            ship.task = GatherEnergy()
            ship.action_queue = path_to_actions(path)
        else:
            ship.task = None


def get_positions_with_max_energy(state, positions):
    position_to_energy = {}
    for x, y in positions:
        node = state.space.get_node(x, y)

        energy = node.energy
        if energy is None:
            continue

        if node.type == NodeType.nebula:
            energy -= Params.NEBULA_ENERGY_REDUCTION

        position_to_energy[(x, y)] = energy

    if not position_to_energy:
        return []

    max_energy = max(position_to_energy.values())

    return [xy for xy, energy in position_to_energy.items() if energy >= max_energy - 1]
