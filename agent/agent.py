from sys import stderr as err

from .base import Params
from .harvest import harvest
from .space import Space
from .fleet import Fleet
from .explore import explore


class State:
    def __init__(self, team_id):
        self.global_step = 0  # global step of the game
        self.match_step = 0  # current step in the match
        self.match_number = 0  # current match number

        self.space = Space()
        self.fleet = Fleet(team_id)
        self.opp_fleet = Fleet(1 - team_id)

    def update(self, obs):
        if obs["steps"] > 0:
            self._update_step_counters()

        assert obs["steps"] == self.global_step
        assert obs["match_steps"] == self.match_step

        if self.match_step == 0:
            self.fleet.clear()
            self.opp_fleet.clear()
            return

        points = int(obs["team_points"][self.fleet.team_id])
        reward = max(0, points - self.fleet.points)

        self.space.update(obs, team_to_reward={self.fleet.team_id: reward})
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

    def _update_step_counters(self):
        self.global_step += 1
        self.match_step += 1
        if self.match_step > Params.MAX_STEPS_IN_MATCH:
            self.match_step = 0
            self.match_number += 1

    def steps_left_in_match(self):
        return Params.MAX_STEPS_IN_MATCH - self.match_step


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        Params.MAX_UNITS = env_cfg["max_units"]
        Params.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Params.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Params.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Params.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.state = State(self.team_id)

    def act(self, step: int, obs, remaining_overage_time: int = 60):
        self.state.update(obs)

        # print(
        #     f"start step {self.state.global_step}"
        #     f", match {self.state.match_number}:{self.state.match_step}",
        #     file=err,
        # )

        space = self.state.space
        fleet = self.state.fleet

        # space.show_map(fleet)
        # space.show_energy_field()
        # space.show_exploration_info()

        if self.state.match_step == 0:
            return fleet.create_actions_array()

        explore(self.state)
        harvest(self.state)

        # fleet.show_tasks()

        return fleet.create_actions_array()
