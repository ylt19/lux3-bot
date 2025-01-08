from .base import nearby_positions, SPACE_SIZE
from .path import find_path_in_dynamic_environment, path_to_actions


def evasion(state):
    direct_targets_for_opp_sap_ships = state.field.direct_targets_for_opp_sap_ships

    for ship in state.fleet:
        x, y = ship.coordinates
        if not direct_targets_for_opp_sap_ships[y, x]:
            continue

        if not ship.can_move():
            continue

        rs = state.grid.resumable_search(ship.unit_id)

        min_distance, best_target = float("inf"), None
        for target in nearby_positions(x, y, SPACE_SIZE):
            if direct_targets_for_opp_sap_ships[target[1], target[0]]:
                continue

            distance = rs.distance(target)
            if distance < min_distance:
                min_distance, best_target = distance, target

        if best_target is None:
            continue

        target_node = state.space.get_node(*best_target)

        path = find_path_in_dynamic_environment(
            state,
            start=ship.coordinates,
            goal=target_node.coordinates,
            ship_energy=ship.energy,
        )

        if path:
            ship.action_queue = path_to_actions(path)
