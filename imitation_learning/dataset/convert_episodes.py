import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from agent.path import ActionType
from agent.base import (
    Global,
    chebyshev_distance,
    manhattan_distance,
    get_nebula_tile_drift_speed,
)
from agent.state import State
from agent.space import NodeType


def convert_episode(episode_data, team_id):
    win_teams = get_win_team_by_match(episode_data)
    wins = [x == team_id for x in win_teams]
    num_wins = sum(wins)

    game_params = episode_data["params"]
    team_actions = np.array(
        [x[f"player_{team_id}"] for x in episode_data["actions"]], dtype=np.int8
    )
    team_observations = [
        convert_episode_step(x, team_id) for x in episode_data["observations"]
    ]

    exploration_flags = estimate_hidden_constant_discovery_steps(
        team_id, team_observations, team_actions, game_params
    )

    agent_episode = {
        "wins": wins,
        "num_wins": num_wins,
        "team_id": team_id,
        "episode_id": episode_data["metadata"]["episode_id"],
        "seed": episode_data["metadata"]["seed"],
        "agent": episode_data["metadata"]["agents"][team_id],
        "opponent": episode_data["metadata"]["agents"][1 - team_id],
        "observations": team_observations,
        "actions": team_actions,
        "params": episode_data["params"],
        "exploration_flags": exploration_flags,
    }

    return agent_episode


def convert_episodes(submission_id):
    dir_ = Path(__file__).parent

    if not os.path.exists(f"{dir_}/agent_episodes"):
        os.mkdir(f"{dir_}/agent_episodes")

    games = pd.read_csv(f"{dir_}/games.csv", usecols=["SubmissionId", "EpisodeId"])
    games = games[games["SubmissionId"] == submission_id]

    episodes = sorted([int(x) for x in games["EpisodeId"].unique()])

    for i, episode_id in enumerate(episodes, start=1):
        print(f"converting {episode_id}: {i}/{len(episodes)}")
        episode_path = f"{dir_}/episodes/{episode_id}.json"
        agent_episode_path = f"{dir_}/agent_episodes/{submission_id}_{episode_id}.pkl"

        if not os.path.exists(episode_path) or os.path.exists(agent_episode_path):
            continue

        episode_data = json.load(open(episode_path, "r"))
        agents = episode_data["metadata"]["agents"]

        for team_id, agent in enumerate(agents):
            if agent["submission_id"] == submission_id:
                agent_episode = convert_episode(episode_data, team_id)
                pickle.dump(agent_episode, open(agent_episode_path, "wb"))


def convert_episode_step(episode_step, team_id):

    sensor_mask = np.array(episode_step["vision_power_map"])[team_id]
    sensor_mask = (sensor_mask > 0).astype(np.int8)

    obs_energy = np.array(episode_step["map_features"]["energy"], dtype=np.int8)
    obs_energy[sensor_mask == 0] = -1

    tile_type = np.array(episode_step["map_features"]["tile_type"], dtype=np.int8)
    tile_type[sensor_mask == 0] = -1

    relic_nodes = []
    relic_nodes_mask = []
    for i in range(6):
        x, y = -1, -1
        is_visible = False
        if i < len(episode_step["relic_nodes"]):
            x_, y_ = episode_step["relic_nodes"][i]
            if sensor_mask[x_, y_]:
                is_visible = True
                x, y = x_, y_

        relic_nodes.append([x, y])
        relic_nodes_mask.append(is_visible)

    units_mask = [[], []]
    units_energy = [[], []]
    units_position = [[], []]
    for team in range(2):
        for i in range(16):
            x, y = -1, -1
            energy = -1
            is_visible = False

            if episode_step["units_mask"][team][i]:
                x_, y_ = episode_step["units"]["position"][team][i]
                if sensor_mask[x_, y_] or team == team_id:
                    is_visible = True
                    x, y = x_, y_
                    energy = episode_step["units"]["energy"][team][i][0]

            units_mask[team].append(is_visible)
            units_energy[team].append(energy)
            units_position[team].append([x, y])

    obs = {
        "steps": episode_step["steps"],
        "match_steps": episode_step["match_steps"],
        "team_wins": episode_step["team_wins"],
        "team_points": episode_step["team_points"],
        "sensor_mask": sensor_mask,
        "relic_nodes": relic_nodes,
        "relic_nodes_mask": relic_nodes_mask,
        "map_features": {"energy": obs_energy, "tile_type": tile_type},
        "units_mask": units_mask,
        "units": {"energy": units_energy, "position": units_position},
    }

    return obs


def get_win_team_by_match(episode_data):
    win_teams = []

    num_matches_in_game = episode_data["params"]["match_count_per_episode"]
    max_steps_in_match = episode_data["params"]["max_steps_in_match"]

    previous_team_wins = [0, 0]
    for match in range(1, num_matches_in_game + 1):
        last_step = match * (max_steps_in_match + 1)
        team_wins = episode_data["observations"][last_step]["team_wins"]

        if team_wins[0] > previous_team_wins[0]:
            win_teams.append(0)
        elif team_wins[1] > previous_team_wins[1]:
            win_teams.append(1)

        previous_team_wins = team_wins

    return win_teams


def estimate_hidden_constant_discovery_steps(
    team_id, team_observations, team_actions, game_params
):
    """
    Estimate the steps at which the agent with the given `team_id`
    discovers certain hidden constants during the game.

    Hidden constants include:
    - nebula_energy_reduction
    - nebula_tile_drift_speed
    - unit_sap_dropoff_factor
    - unit_energy_void_factor

    Returns:
    - A dictionary {constant name -> estimated step when the constant was discovered}
    """

    inf_step = (
        game_params["max_steps_in_match"] * (game_params["match_count_per_episode"] + 1)
        + 1
    )

    nebula_tile_drift_speed_found_step = inf_step
    unit_sap_dropoff_factor_found_step = inf_step
    nebula_energy_reduction_found_step = inf_step
    unit_energy_void_factor_found_step = inf_step

    Global.clear()
    Global.VERBOSITY = 1

    Global.MAX_UNITS = game_params["max_units"]
    Global.UNIT_MOVE_COST = game_params["unit_move_cost"]
    Global.UNIT_SAP_COST = game_params["unit_sap_cost"]
    Global.UNIT_SAP_RANGE = game_params["unit_sap_range"]
    Global.UNIT_SENSOR_RANGE = game_params["unit_sensor_range"]

    state = State(team_id)

    previous_step_sap_positions = set()
    previous_step_opp_ships = set()
    for obs, actions in zip(team_observations, team_actions):
        state.update(obs)
        step = state.global_step

        if (
            nebula_tile_drift_speed_found_step == inf_step
            and Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        ):
            if (
                abs(
                    get_nebula_tile_drift_speed()
                    - game_params["nebula_tile_drift_speed"]
                )
                > 0.001
            ):
                raise RuntimeError(
                    f"nebula_tile_drift_speed is wrong, params={game_params['nebula_tile_drift_speed']}, "
                    f"found={get_nebula_tile_drift_speed()}"
                )
            nebula_tile_drift_speed_found_step = step

        if (
            Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and Global.UNIT_SAP_DROPOFF_FACTOR_FOUND
            and Global.NEBULA_ENERGY_REDUCTION_FOUND
            and Global.UNIT_ENERGY_VOID_FACTOR_FOUND
        ):
            # we have found all the hidden constants
            break

        if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
            # we assume that we have found a constant when the first ship enters Nebula
            for ship in state.fleet:
                if ship.node.type == NodeType.nebula:
                    Global.NEBULA_ENERGY_REDUCTION_FOUND = True
                    Global.NEBULA_ENERGY_REDUCTION = game_params[
                        "nebula_tile_energy_reduction"
                    ]
                    nebula_energy_reduction_found_step = step
                    break

        if not Global.UNIT_SAP_DROPOFF_FACTOR_FOUND:
            #  we assume that we found the constant when we made the first sap that reached the opponent's ship.

            for sap_position in previous_step_sap_positions:
                for opp_uint_id in previous_step_opp_ships:
                    opp_ship = state.opp_fleet.ships[opp_uint_id]
                    if opp_ship.node is None:
                        continue
                    distance = chebyshev_distance(opp_ship.coordinates, sap_position)
                    if distance == 1:
                        Global.UNIT_SAP_DROPOFF_FACTOR_FOUND = True
                        Global.UNIT_SAP_DROPOFF_FACTOR = game_params[
                            "unit_sap_dropoff_factor"
                        ]
                        unit_sap_dropoff_factor_found_step = step
                        break

            previous_step_sap_positions = set()
            for ship, (action_type, dx, dy) in zip(state.fleet.ships, actions):
                if (
                    action_type == ActionType.sap
                    and ship.node is not None
                    and ship.can_sap()
                ):
                    previous_step_sap_positions.add(
                        (ship.node.x + dx, ship.node.y + dy)
                    )

            previous_step_opp_ships = set()
            for opp_ship in state.opp_fleet:
                if opp_ship.node is not None and opp_ship.energy > 0:
                    previous_step_opp_ships.add(opp_ship.unit_id)

        if not Global.UNIT_ENERGY_VOID_FACTOR_FOUND:
            # we assume that we have found a constant when our ships are next to an enemy ship for the first time.
            for opp_ship in state.opp_fleet:
                for our_ship in state.fleet:
                    dinstance = manhattan_distance(
                        opp_ship.coordinates, our_ship.coordinates
                    )
                    if dinstance == 1:
                        Global.UNIT_ENERGY_VOID_FACTOR_FOUND = True
                        Global.UNIT_ENERGY_VOID_FACTOR = game_params[
                            "unit_energy_void_factor"
                        ]
                        unit_energy_void_factor_found_step = step
                        break

    return {
        "nebula_energy_reduction": nebula_energy_reduction_found_step,
        "nebula_tile_drift_speed": nebula_tile_drift_speed_found_step,
        "unit_sap_dropoff_factor": unit_sap_dropoff_factor_found_step,
        "unit_energy_void_factor": unit_energy_void_factor_found_step,
    }
