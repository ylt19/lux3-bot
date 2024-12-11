from .base import Params, log
from .state import State

from .harvest import harvest
from .explore import explore
from .gather_energy import gather_energy


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        Params.MAX_UNITS = env_cfg["max_units"]
        Params.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Params.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Params.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Params.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.state = State(self.team_id)
        self.previous_state = self.state.copy()

    def act(self, step: int, obs, remaining_overage_time: int = 60):
        self.state.update(obs)

        log(
            f"start step {self.state.global_step}"
            f", match {self.state.match_number}:{self.state.match_step}"
            f", wins {Params.NUM_WINS}/{Params.NUM_COMPLETED_MATCHES}"
        )

        if self.state.match_step == 0:
            self.previous_state = self.state.copy()

        # self.state.show_visible_map()
        # self.state.show_visible_energy_field()
        # self.state.show_explored_map()
        # self.state.show_explored_energy_field()
        # self.state.show_exploration_info()

        explore(self.previous_state, self.state)
        harvest(self.state)
        gather_energy(self.state)

        # self.state.show_tasks(show_path=False)

        self.previous_state = self.state.copy()
        return self.state.create_actions_array()
