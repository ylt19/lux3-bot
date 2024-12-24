from .base import (
    log,
    Task,
    Global,
    is_team_sector,
    nearby_positions,
    get_spawn_location,
    manhattan_distance,
    SECTOR_SIZE,
)
from .path import (
    path_to_actions,
    estimate_energy_cost,
    find_path_in_dynamic_environment,
)
from .space import Node


class RelicFinder(Task):

    def __repr__(self):
        return f"{self.__class__.__name__}{self.target.coordinates}"

    def completed(self, state):
        return True  # self.target.explored_for_relic

    @classmethod
    def generate_tasks(cls, state):
        targets = get_targets_for_relic_exploration(state)
        tasks = []
        for target in targets:
            tasks.append(cls(state.space.get_node(*target)))
        return tasks

    def evaluate(self, state, ship):
        if not ship.can_move():
            return 0

        rs = state.get_resumable_dijkstra(ship.unit_id)
        path = rs.find_path(self.target.coordinates)
        energy_needed = estimate_energy_cost(state.space, path)
        grid_distance = rs.distance(self.target.coordinates)

        energy_remain = ship.energy - energy_needed

        score = 1000 - grid_distance + energy_remain

        return score

    def apply(self, state, ship):
        path = find_path_in_dynamic_environment(
            state,
            start=ship.coordinates,
            goal=self.target.coordinates,
            ship_energy=ship.energy,
        )

        ship.action_queue = path_to_actions(path)


class VoidSeeker(Task):

    def __init__(self, relic_node, target=None):
        self.relic_node = relic_node
        super().__init__(target)

    def __repr__(self):
        target = self.target.coordinates if self.target else None
        return f"{self.__class__.__name__}(relic={self.relic_node.coordinates}, target={target})"

    def completed(self, state):
        return is_relic_fully_explored(state, self.target)

    @classmethod
    def generate_tasks(cls, state):
        relic_nodes = set(get_unexplored_relics(state))
        for ship in state.fleet:
            if isinstance(ship.task, VoidSeeker):
                relic_node = ship.task.relic_node
                if relic_node in relic_nodes:
                    relic_nodes.remove(relic_node)

        tasks = []
        for node in relic_nodes:
            tasks.append(VoidSeeker(relic_node=node))

        return tasks

    def evaluate(self, state, ship):
        if not ship.can_move():
            return 0

        rs = state.get_resumable_dijkstra(ship.unit_id)

        target_node, min_distance = None, float("inf")
        for node in get_unexplored_for_reward_nodes(state, self.relic_node):
            grid_distance = rs.distance(node.coordinates)
            if grid_distance < min_distance:
                target_node, min_distance = node, grid_distance

        if target_node is None:
            return 0

        path = rs.find_path(target_node.coordinates)
        energy_needed = estimate_energy_cost(state.space, path)
        grid_distance = rs.distance(target_node.coordinates)

        energy_remain = ship.energy - energy_needed

        score = 1200 - grid_distance + energy_remain

        return score

    def apply(self, state, ship):

        rs = state.get_resumable_dijkstra(ship.unit_id)

        target_node, min_distance = None, float("inf")
        for node in get_unexplored_for_reward_nodes(state, self.relic_node):
            grid_distance = rs.distance(node.coordinates)
            if grid_distance < min_distance:
                target_node, min_distance = node, grid_distance

        if not target_node:
            return

        if ship.node == target_node:
            go_to_known_node = False
            for other_ship in state.fleet:
                if (
                    other_ship == ship
                    or not isinstance(other_ship.task, VoidSeeker)
                    or not other_ship.action_queue
                ):
                    continue

                next_position = other_ship.next_position()
                if not state.space.get_node(*next_position).explored_for_reward:
                    go_to_known_node = True
                    break

            if go_to_known_node:
                log("go_to_known_node")
                target_node, min_distance = None, float("inf")
                for node in state.space:
                    if node.explored_for_reward:
                        grid_distance = rs.distance(node.coordinates)
                        if grid_distance < min_distance:
                            target_node, min_distance = node, grid_distance

        path = find_path_in_dynamic_environment(
            state,
            start=ship.coordinates,
            goal=target_node.coordinates,
            ship_energy=ship.energy,
        )

        self.target = target_node
        ship.action_queue = path_to_actions(path)


def is_relic_fully_explored(state, relic_node):
    for x, y in nearby_positions(*relic_node.coordinates, Global.RELIC_REWARD_RANGE):
        node = state.space.get_node(x, y)
        if not node.explored_for_reward:
            return False

    return True


def get_unexplored_for_reward_nodes(state, relic_node):
    for x, y in nearby_positions(*relic_node.coordinates, Global.RELIC_REWARD_RANGE):
        node = state.space.get_node(x, y)
        if not node.explored_for_reward:
            yield node


def get_unexplored_relics(state) -> list[Node]:
    relic_nodes = []
    for relic_node in state.space.relic_nodes:
        if is_team_sector(state.team_id, *relic_node.coordinates):
            if not is_relic_fully_explored(state, relic_node):
                relic_nodes.append(relic_node)

    return relic_nodes


def get_targets_for_relic_exploration(state, max_num_targets=16):
    space = state.space
    sensor_range = Global.UNIT_SENSOR_RANGE
    spawn_location = get_spawn_location(state.team_id)

    def spawn_distance(t_):
        return manhattan_distance(spawn_location, t_)

    targets = []
    for node in space:
        x, y = node.coordinates
        if not node.explored_for_relic and is_team_sector(state.team_id, x, y):
            targets.append((x, y))

    if len(targets) > SECTOR_SIZE / 2:
        # First of all, we want to explore the area in the center of the map.
        targets = [
            (x, y)
            for x, y in targets
            if (
                x >= sensor_range
                and y >= sensor_range
                and Global.SPACE_SIZE - x - 1 >= sensor_range
                and Global.SPACE_SIZE - y - 1 >= sensor_range
            )
        ]

    targets = sorted(targets, key=spawn_distance)

    explored = set()
    filtered_targets = []
    for target in targets:
        if target in explored:
            continue

        filtered_targets.append(target)
        if len(filtered_targets) == max_num_targets:
            break

        for xy in nearby_positions(*target, sensor_range):
            explored.add(xy)

    return filtered_targets
