import numpy as np
from scipy.signal import convolve2d

from .base import Task, Global, SPACE_SIZE, manhattan_distance
from .path import (
    estimate_energy_cost,
    path_to_actions,
    find_path_in_dynamic_environment,
)


class Control(Task):

    def __init__(self, target, energy_gain):
        super().__init__(target)
        self.energy_gain = energy_gain

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target.coordinates, self.energy_gain})"

    @classmethod
    def generate_tasks(cls, state):
        if not state.space.reward_nodes:
            return []

        control_positions = find_control_points(state)

        tasks = []
        for x, y in control_positions:
            vision_gain = state.field.vision_gain[y, x]
            if vision_gain < 9:
                continue

            min_reward_distance = min(
                manhattan_distance(node.coordinates, (x, y))
                for node in state.space.reward_nodes
            )

            if min_reward_distance > 10:
                continue

            target = state.space.get_node(x, y)
            tasks.append(Control(target, target.energy_gain))

        return tasks

    def evaluate(self, state, ship):
        # if ship.energy < 100:
        #     return 0

        rs = state.get_resumable_dijkstra(ship.unit_id)
        path = rs.find_path(self.target.coordinates)
        energy_needed = estimate_energy_cost(state.space, path)

        return 700 + (-5) * len(path) + (-0.2) * energy_needed

    def completed(self, state, ship):
        if self.target.energy_gain != self.energy_gain:
            return True
        return False

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


def find_control_points(state):
    vision_by_ship = 5  # 2 * Global.UNIT_SENSOR_RANGE + 1
    min_vision_gain = 15

    vision_kernel = np.ones((vision_by_ship, vision_by_ship), dtype=np.float32)

    control = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)

    for ship in state.fleet:
        if isinstance(ship.task, Control):
            target = ship.task.target
            x, y = target.coordinates

            ship_control = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
            ship_control[y, x] = 1

            ship_control = convolve2d(
                ship_control,
                vision_kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )

            control = np.logical_or(control, ship_control)

    energy_gain = state.field.energy_gain

    positions = []

    while True:

        vision_gain = convolve2d(
            (control == 0),
            vision_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        max_gain = vision_gain.max()
        if max_gain < min_vision_gain:
            break

        energy_left = np.array(energy_gain)
        energy_left[np.where(vision_gain < min_vision_gain)] = (
            Global.MIN_ENERGY_PER_TILE - 1
        )

        # show_field(energy_left)

        max_energy = energy_left.max()
        if max_energy < 0:
            break

        yy, xx = np.where(energy_left == max_energy)
        y, x = int(yy[0]), int(xx[0])

        positions.append((x, y))

        # print(best_position, max_energy)

        ship_control = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        ship_control[y, x] = 1

        ship_control = convolve2d(
            ship_control,
            vision_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        control = np.logical_or(control, ship_control)

    return positions
