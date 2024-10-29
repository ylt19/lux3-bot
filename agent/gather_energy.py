from pathfinding import ResumableBFS

from .base import Params
from .path import (
    path_to_actions,
    find_closest_target,
    estimate_energy_cost,
    find_path_in_dynamic_environment,
)
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

    grid = state.obstacle_grid
    rs = ResumableBFS(grid, (0, 0))
    components = [set(x) for x in grid.find_components()]
    steps_left = state.steps_left_in_match()

    for ship in state.fleet:
        ship_position = ship.coordinates

        if ship.task and not isinstance(ship.task, GatherEnergy):
            continue

        available_locations = {ship.coordinates}
        for component in components:
            if ship_position in component:
                available_locations = component
                break

        rs.start_node = ship_position
        available_locations = [
            xy for xy in available_locations if rs.distance(xy) < steps_left
        ]

        targets = get_positions_with_max_energy(state, available_locations)

        target, _ = find_closest_target(state, ship.coordinates, targets)
        if not target:
            ship.task = None
            continue

        path = find_path_in_dynamic_environment(
            state, start=ship.coordinates, goal=target, ship_energy=ship.energy
        )
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
