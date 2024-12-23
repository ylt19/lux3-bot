from .base import Global, log
from .state import State

from .fleet import find_hidden_constants
from .tasks import apply_tasks


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.state = State(self.team_id)
        self.previous_state = self.state.copy()

    def act(self, step: int, obs, remaining_overage_time: int = 60):
        state = self.state

        state.update(obs)

        log(
            f"start step {state.global_step:>3}"
            f", match {state.match_number}:{state.match_step:>3}"
            f", wins {Global.NUM_WINS}/{Global.NUM_COMPLETED_MATCHES}"
            f", points {state.fleet.points}:{state.opp_fleet.points}"
        )

        if state.match_step == 0:
            self.previous_state = state.copy()

        find_hidden_constants(self.previous_state, state)

        # state.show_visible_map()
        # state.show_visible_energy_field()
        # state.show_explored_map()
        # state.show_explored_energy_field()
        # state.show_exploration_map()

        apply_tasks(state)
        # state.show_tasks(show_path=False)

        self.previous_state = state.copy()
        return state.create_actions_array()
