from sys import stderr as err

from .base import Params, is_team_sector, nearby_positions
from .path import PathFinder, path_to_actions, estimate_energy_cost
from .space import Node, NodeType
from .tasks import FindRelicNodes, FindRewardNodes, GatherEnergy


def explore(previous_state, state):
    find_nebula_energy_reduction(previous_state, state)
    find_relics(state)
    find_rewards(state)


def find_relics(state):
    space = state.space
    fleet = state.fleet

    if Params.ALL_RELICS_FOUND:
        delete_tasks(fleet, FindRelicNodes)
        return

    targets = set()
    for node in space:
        if not node.explored_for_relic and is_team_sector(
            fleet.team_id, *node.coordinates
        ):
            targets.add(node.coordinates)

    finder = PathFinder(state)

    for ship in fleet:
        if ship.task and not isinstance(ship.task, (FindRelicNodes, GatherEnergy)):
            continue

        if not targets:
            ship.task = None
            continue

        target, _ = finder.find_closest_target(ship.coordinates, targets)
        if not target:
            ship.task = None
            continue

        path = finder.find_path(ship.coordinates, target, dynamic=True)
        energy = estimate_energy_cost(state.space, path)

        if ship.energy >= energy:
            ship.task = FindRelicNodes()
            ship.action_queue = path_to_actions(path)

            for x, y in path:
                for xy in nearby_positions(x, y, Params.UNIT_SENSOR_RANGE):
                    if xy in targets:
                        targets.remove(xy)
        else:
            ship.task = None


def find_rewards(state):
    space = state.space
    fleet = state.fleet

    if Params.ALL_REWARDS_FOUND:
        delete_tasks(fleet, FindRewardNodes)
        return

    relic_nodes = get_unexplored_relics(space, fleet.team_id)

    relic_node_to_ship = {}
    for ship in fleet:
        if not isinstance(ship.task, FindRewardNodes):
            continue

        relic_node = space.get_node(*ship.task.coordinates)
        if relic_node not in relic_nodes or ship.energy < Params.UNIT_MOVE_COST * 10:
            ship.task = None
            ship.action_queue = []
            continue

        relic_node_to_ship[relic_node] = ship

    finder = PathFinder(state)

    for relic in relic_nodes:
        if relic not in relic_node_to_ship:
            ship = find_ship_for_reward_task(space, fleet, relic, finder)
            if ship:
                relic_node_to_ship[relic] = ship

    relic_ships = sorted(list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id)

    pause_action = False
    for relic_node, ship in relic_ships:

        targets = []
        for x, y in nearby_positions(
            *relic_node.coordinates, Params.RELIC_REWARD_RANGE
        ):
            node = space.get_node(x, y)
            if not node.explored_for_reward:
                targets.append((x, y))

        target, _ = finder.find_closest_target(ship.coordinates, targets)

        if target == ship.coordinates:
            if not pause_action:
                pause_action = True
            else:
                target, _ = finder.find_closest_target(
                    ship.coordinates,
                    targets=[n.coordinates for n in space if n.explored_for_reward],
                )

        if not target:
            continue

        path = finder.find_path(ship.coordinates, target, dynamic=True)
        energy = estimate_energy_cost(state.space, path)
        if ship.energy >= energy:
            ship.action_queue = path_to_actions(path)
            ship.task = FindRewardNodes(relic_node)
        else:
            if isinstance(ship.task, FindRewardNodes):
                ship.task = None


def get_unexplored_relics(space, team_id) -> list[Node]:
    relic_nodes = []
    for relic_node in space.relic_nodes:
        if not is_team_sector(team_id, *relic_node.coordinates):
            continue

        explored = True
        for x, y in nearby_positions(
            *relic_node.coordinates, Params.RELIC_REWARD_RANGE
        ):
            node = space.get_node(x, y)
            if not node.explored_for_reward and node.is_walkable:
                explored = False
                break

        if explored:
            continue

        relic_nodes.append(relic_node)

    return relic_nodes


def find_ship_for_reward_task(space, fleet, relic_node, finder):
    free_ships = []
    for ship in fleet:
        if isinstance(ship.task, FindRewardNodes):
            continue
        if ship.energy < Params.UNIT_MOVE_COST * 10:
            continue
        free_ships.append(ship)

    if not free_ships:
        return

    unexplored = []
    for x, y in nearby_positions(*relic_node.coordinates, Params.RELIC_REWARD_RANGE):
        node = space.get_node(x, y)
        if not node.explored_for_reward:
            unexplored.append((x, y))

    closest_ship, min_distance = None, float("inf")
    for ship in free_ships:
        _, distance = finder.find_closest_target(ship.coordinates, unexplored)
        if distance < min_distance:
            closest_ship, min_distance = ship, distance

    return closest_ship


def delete_tasks(fleet, task_type):
    for ship in fleet:
        if isinstance(ship.task, task_type):
            ship.task = None


def find_nebula_energy_reduction(previous_state, state):
    if Params.NEBULA_ENERGY_REDUCTION_FOUND:
        return

    for previous_ship, ship in zip(previous_state.fleet, state.fleet):
        if previous_ship.node is None or ship.node is None:
            continue

        node = ship.node
        if node.energy is None:
            continue

        is_moving = int(node != previous_ship.node)

        if previous_ship.energy < 30 - Params.UNIT_MOVE_COST * is_moving:
            continue

        if (
            node.type != NodeType.nebula
            or previous_state.space.get_node(*node.coordinates).type != NodeType.nebula
        ):
            continue

        delta = (
            previous_ship.energy
            - ship.energy
            + node.energy
            - Params.UNIT_MOVE_COST * is_moving
        )

        # print(previous_ship.node, "->", node, "delta", delta, file=err)

        if delta > 20:
            Params.NEBULA_ENERGY_REDUCTION = 100
        elif abs(delta - 10) < 5:
            Params.NEBULA_ENERGY_REDUCTION = 10
        elif abs(delta - 0) < 5:
            Params.NEBULA_ENERGY_REDUCTION = 0
        else:
            continue

        Params.NEBULA_ENERGY_REDUCTION_FOUND = True

        print(
            f"Find param NEBULA_ENERGY_REDUCTION = {Params.NEBULA_ENERGY_REDUCTION}",
            file=err,
        )
        return
