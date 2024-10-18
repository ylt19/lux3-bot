from sys import stderr as err
import numpy as np

from .base import Step
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

        self.step = Step()
        self.space = Space()
        self.fleet = Fleet(self.team_id, env_cfg)
        self.opp_fleet = Fleet(self.opp_team_id, env_cfg)

    def act(self, global_step: int, obs, remaining_overage_time: int = 60):
        if global_step > 0:
            self.step.update()

        # print(f"start {self.step}", file=err)

        if self.step.match_step == 0:
            self.fleet.clear()
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

        explore(self.step, self.space, self.fleet)
        harvest(self.step, self.space, self.fleet)

        # self.fleet.show_tasks()

        return self.fleet.create_actions_array()
