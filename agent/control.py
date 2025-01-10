from .base import Task, Global, SPACE_SIZE, manhattan_distance
from .path import (
    estimate_energy_cost,
    path_to_actions,
    find_path_in_dynamic_environment,
)


class Control(Task):

    def __init__(self, target):
        super().__init__(target)

    def __repr__(self):
        if self.reward:
            return f"Reward({self.target.coordinates})"
        return f"Control({self.target.coordinates})"

    @property
    def reward(self):
        return self.target.reward

    @classmethod
    def generate_tasks(cls, state):
        if not state.space.reward_nodes:
            return []

        control_positions = set(
            state.field.reward_positions + state.field.control_positions
        )

        for ship in state.fleet:
            if isinstance(ship.task, Control):
                p = ship.task.target.coordinates
                if p in control_positions:
                    control_positions.remove(p)

        tasks = []
        for x, y in control_positions:

            min_reward_distance = min(
                manhattan_distance(node.coordinates, (x, y))
                for node in state.space.reward_nodes
            )

            if min_reward_distance > 10:
                continue

            target = state.space.get_node(x, y)
            tasks.append(Control(target))

        return tasks

    def evaluate(self, state, ship):
        if not ship.can_move():
            if ship.node == self.target:
                return 1000
            else:
                return 0

        if ship.energy < state.field.opp_protection[self.target.y, self.target.x]:
            return 0

        rs = state.grid.resumable_search(ship.unit_id)
        path = rs.find_path(self.target.coordinates)
        if not path:
            return 0
        if len(path) > state.steps_left_in_match():
            return 0

        energy_needed = estimate_energy_cost(state.space, path)
        spawn_distance = state.fleet.spawn_distance(*self.target.coordinates)
        middle_lane_distance = max(spawn_distance - SPACE_SIZE, 0)

        p = Global.Params
        score = (
            p.CONTROL_INIT_SCORE
            + p.CONTROL_REWARD_SCORE * self.reward
            + p.CONTROL_PATH_LENGTH_MULTIPLIER * len(path)
            + p.CONTROL_ENERGY_COST_MULTIPLIER * energy_needed
            + p.CONTROL_NODE_ENERGY_MULTIPLIER * self.target.energy_gain
            + p.CONTROL_MIDDLE_LANE_DISTANCE_MULTIPLIER * middle_lane_distance
        )
        return score

    def completed(self, state, ship):
        if ship.energy < state.field.opp_protection[self.target.y, self.target.x]:
            return True
        if not ship.can_move() and ship.node != self.target:
            return True

        p = self.target.coordinates
        if (
            p not in state.field.reward_positions
            and p not in state.field.control_positions
        ):
            return True
        return False

    def apply(self, state, ship):
        rs = state.grid.resumable_search(ship.unit_id)

        path = rs.find_path(self.target.coordinates)
        if not path:
            return False

        energy_needed = estimate_energy_cost(state.space, path)
        if energy_needed > ship.energy:
            path = find_path_in_dynamic_environment(
                state,
                start=ship.coordinates,
                goal=self.target.coordinates,
                ship_energy=ship.energy,
                grid=state.grid.energy_with_low_ground,
            )

        ship.action_queue = path_to_actions(path)
        return True
