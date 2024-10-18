from sys import stderr as err
import numpy as np

from .base import Params
from .harvest import harvest
from .space import Space
from .fleet import Fleet
from .explore import explore


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.global_step = -1  # global step of the game
        self.match_step = -1  # current step in the match
        self.match_number = 0  # current match number

        Params.MAX_UNITS = env_cfg["max_units"]
        Params.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Params.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Params.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Params.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

    def act(self, step: int, obs, remaining_overage_time: int = 60):
        self.update_step_counters()
        assert step == self.global_step
        assert obs["match_steps"] == self.match_step

        # print(
        #     f"start step {self.global_step}, match {self.match_number}:{self.match_step}",
        #     file=err,
        # )

        if self.match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            return self.fleet.create_actions_array()

        num_points = int(obs["team_points"][self.team_id])
        reward = max(0, num_points - self.fleet.points)

        self.space.update(obs, team_to_reward={self.team_id: reward})
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        # self.space.show_relic_info()
        # self.space.show_map(self.fleet)
        # self.space.show_energy_field()
        # self.space.show_exploration_info()

        explore(self)
        harvest(self)

        # self.fleet.show_tasks()

        return self.fleet.create_actions_array()

    def update_step_counters(self):
        self.global_step += 1
        self.match_step += 1
        if self.match_step > Params.MAX_STEPS_IN_MATCH:
            self.match_step = 0
            self.match_number += 1

    def steps_left_in_match(self):
        return Params.MAX_STEPS_IN_MATCH - self.match_step
