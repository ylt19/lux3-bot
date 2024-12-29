import numpy as np
from scipy.signal import convolve2d

from .base import (
    log,
    Global,
    SPACE_SIZE,
    is_team_sector,
    manhattan_distance,
    get_spawn_location,
)
from .path import (
    path_to_actions,
    find_path_in_dynamic_environment,
)
from .base import Task


class Heal(Task):

    def __init__(self, ship, target=None):
        super().__init__(target)
        self.ship = ship

    def __repr__(self):
        if self.target is None:
            return self.__class__.__name__
        else:
            coordinates = self.target.coordinates
            energy = self.target.energy_gain
            return f"{self.__class__.__name__}(target={coordinates}, energy={energy})"

    def completed(self, state, ship):
        return True

    @classmethod
    def generate_tasks(cls, state):
        return [Heal(ship) for ship in state.fleet]

    def evaluate(self, state, _):
        ship = self.ship

        opp_spawn_location = get_spawn_location(state.opp_team_id)
        opp_spawn_distance = manhattan_distance(opp_spawn_location, ship.coordinates)
        ship_energy = ship.energy

        p = Global.Params
        score = (
            p.HEAL_INIT_SCORE
            + opp_spawn_distance * p.HEAL_OPP_SPAWN_DISTANCE_MULTIPLIER
            + ship_energy * p.HEAL_SHIP_ENERGY_MULTIPLIER
        )

        return score

    def apply(self, state, _):
        target = self.find_target(state)
        if not target:
            return False

        ship = self.ship
        path = find_path_in_dynamic_environment(
            state,
            start=ship.coordinates,
            goal=target.coordinates,
            ship_energy=ship.energy,
        )
        if not path:
            return False

        self.target = target
        ship.action_queue = path_to_actions(path)
        return True

    def find_target(self, state):
        ship = self.ship

        rs = state.get_resumable_dijkstra(ship.unit_id)
        steps_left_in_match = state.steps_left_in_match()

        node_to_score = {}
        if Global.Params.HEAL_NEAR_REWARDS:

            reward_nodes = [
                x
                for x in state.space.reward_nodes
                if is_team_sector(state.team_id, *x.coordinates)
            ]

            reward_array = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
            for node in reward_nodes:
                reward_array[node.y, node.x] = 1

            reward_array = convolve2d(
                reward_array,
                np.ones((Global.UNIT_SAP_RANGE, Global.UNIT_SAP_RANGE), dtype=np.int32),
                mode="same",
                boundary="fill",
                fillvalue=0,
            )

            for x in range(SPACE_SIZE):
                for y in range(SPACE_SIZE):
                    path = rs.find_path((x, y))
                    if (
                        reward_array[y, x] > 0
                        and path
                        and len(path) < steps_left_in_match
                    ):
                        node = state.space.get_node(x, y)
                        node_to_score[node] = node.energy_gain

        if not node_to_score:
            for x in range(SPACE_SIZE):
                for y in range(SPACE_SIZE):
                    path = rs.find_path((x, y))
                    if path and len(path) < steps_left_in_match:
                        node = state.space.get_node(x, y)
                        node_to_score[node] = node.energy_gain

        if not node_to_score:
            return

        best_score = max(node_to_score.values())

        closest_target, min_distance = None, float("inf")
        for node, score in node_to_score.items():
            if score >= best_score - 1:
                distance = rs.distance((node.x, node.y))
                if distance < min_distance:
                    closest_target, min_distance = node, distance

        return closest_target
