import copy

from .base import Params
from .space import Space
from .fleet import Fleet


class State:
    def __init__(self, team_id):
        self.team_id = team_id
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
            self.space.clear()
            return

        points = int(obs["team_points"][self.team_id])
        reward = max(0, points - self.fleet.points)

        self.space.update(obs, team_to_reward={self.team_id: reward})
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

    def _update_step_counters(self):
        self.global_step += 1
        self.match_step += 1
        if self.match_step > Params.MAX_STEPS_IN_MATCH:
            self.match_step = 0
            self.match_number += 1

    def steps_left_in_match(self) -> int:
        return Params.MAX_STEPS_IN_MATCH - self.match_step

    def copy(self) -> "State":
        return copy.deepcopy(self)
