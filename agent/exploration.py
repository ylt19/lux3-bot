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

    def completed(self, state, ship):
        if not ship.can_move():
            return True
        return self.target.explored_for_relic

    @classmethod
    def generate_tasks(cls, state):
        targets = get_targets_for_relic_exploration(
            state, max_num_targets=Global.Params.RELIC_FINDER_NUM_TASKS
        )
        tasks = []
        for target in targets:
            tasks.append(cls(state.space.get_node(*target)))
        return tasks

    def evaluate(self, state, ship):
        if not ship.can_move():
            return 0

        rs = state.grid.resumable_search(ship.unit_id)
        path = rs.find_path(self.target.coordinates)
        if not path:
            return 0
        if len(path) > state.steps_left_in_match():
            return 0

        energy_needed = estimate_energy_cost(state.space, path)

        p = Global.Params
        score = (
            p.RELIC_FINDER_INIT_SCORE
            + p.RELIC_FINDER_PATH_LENGTH_MULTIPLIER * len(path)
            + p.RELIC_FINDER_ENERGY_COST_MULTIPLIER * energy_needed
        )

        return score

    def apply(self, state, ship):
        path = find_path_in_dynamic_environment(
            state,
            start=ship.coordinates,
            goal=self.target.coordinates,
            ship_energy=ship.energy,
        )
        if not path:
            return False

        ship.action_queue = path_to_actions(path)
        return True


class VoidSeeker(Task):

    def __init__(self, relic_node, target=None):
        self.relic_node = relic_node
        super().__init__(target)

    def __repr__(self):
        target = self.target.coordinates if self.target else None
        return f"{self.__class__.__name__}(relic={self.relic_node.coordinates}, target={target})"

    def completed(self, state, ship):
        if not ship.can_move():
            return True
        if self.target is None and is_relic_fully_explored(state, self.relic_node):
            return True
        if self.target.explored_for_reward:
            return True
        return False

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

        rs = state.grid.resumable_search(ship.unit_id)

        target_node, min_distance = None, float("inf")
        for node in get_unexplored_for_reward_nodes(state, self.relic_node):
            path = rs.find_path(node.coordinates)
            if len(path) > state.steps_left_in_match():
                continue
            grid_distance = rs.distance(node.coordinates)
            if grid_distance < min_distance:
                target_node, min_distance = node, grid_distance

        if target_node is None:
            return 0

        path = rs.find_path(target_node.coordinates)
        if not path:
            return 0

        energy_needed = estimate_energy_cost(state.space, path)

        p = Global.Params
        score = (
            p.VOID_SEEKER_INIT_SCORE
            + p.VOID_SEEKER_PATH_LENGTH_MULTIPLIER * len(path)
            + p.VOID_SEEKER_ENERGY_COST_MULTIPLIER * energy_needed
        )

        return score

    def apply(self, state, ship):

        rs = state.grid.resumable_search(ship.unit_id)

        target_node, min_distance = None, float("inf")
        for node in get_unexplored_for_reward_nodes(state, self.relic_node):
            grid_distance = rs.distance(node.coordinates)
            if grid_distance < min_distance:
                target_node, min_distance = node, grid_distance

        if not target_node:
            return False

        path = find_path_in_dynamic_environment(
            state,
            start=ship.coordinates,
            goal=target_node.coordinates,
            ship_energy=ship.energy,
        )

        if len(path) == 0:
            return False

        if len(path) == 1:
            next_node = target_node
        else:
            next_node = state.space.get_node(*path[1])

        if not next_node.explored_for_reward:

            allowed = True
            for other_ship in state.fleet:
                if (
                    other_ship == ship
                    or not isinstance(other_ship.task, VoidSeeker)
                    or not other_ship.action_queue
                ):
                    continue

                next_position = other_ship.next_position()
                if not state.space.get_node(*next_position).explored_for_reward:
                    # the other ship also have an unexplored node in its path
                    # we are blocking the ship's path
                    # it will help generate more useful data in Global.REWARD_RESULTS.
                    allowed = False
                    break

            if not allowed:
                # move to the closest explored node
                target_node, min_distance = None, float("inf")
                for node in state.space:
                    if node.explored_for_reward and node.is_walkable:
                        grid_distance = rs.distance(node.coordinates)

                        # we prefer to find a node with a reward.
                        if node.reward:
                            grid_distance -= 0.01

                        if grid_distance < min_distance:
                            target_node, min_distance = node, grid_distance

                # find a new path
                path = find_path_in_dynamic_environment(
                    state,
                    start=ship.coordinates,
                    goal=target_node.coordinates,
                    ship_energy=ship.energy,
                )

        self.target = target_node
        ship.action_queue = path_to_actions(path)
        return True


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

    targets = set()
    for node in space:
        x, y = node.coordinates
        if not node.explored_for_relic and is_team_sector(state.team_id, x, y):
            targets.add((x, y))

    num_targets = 0
    for ship in state.fleet:
        if isinstance(ship.task, RelicFinder):
            num_targets += 1
            target_coordinates = ship.task.target.coordinates
            for xy in nearby_positions(*target_coordinates, sensor_range):
                if xy in targets:
                    targets.remove(xy)

    def spawn_distance(t_):
        return manhattan_distance(spawn_location, t_)

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
        if len(filtered_targets) == max_num_targets - num_targets:
            break

        for xy in nearby_positions(*target, sensor_range):
            explored.add(xy)

    return filtered_targets
