import os
import sys

import json
import tyro
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

WORKING_FOLDER = Path(__file__).parent
BOT_DIR = WORKING_FOLDER.parent.parent
sys.path.append(str(BOT_DIR))

from agent.path import Action, ActionType
from agent.base import (
    Global,
    SPACE_SIZE,
    transpose,
    get_opposite,
    nearby_positions,
    manhattan_distance,
    chebyshev_distance,
    get_nebula_tile_drift_speed,
)
from agent.space import NodeType
from agent.state import State

EPISODES_DIR = f"{WORKING_FOLDER}/episodes"
OUTPUT_DIR = f"{WORKING_FOLDER}/agent_episodes"
SUBMISSIONS_PATH = f"{WORKING_FOLDER}/submissions.csv"
GAMES_PATH = f"{WORKING_FOLDER}/games.csv"


def convert_episode(episode_data, team_id):
    win_teams = get_win_team_by_match(episode_data)
    wins = [x == team_id for x in win_teams]
    if not any(wins):
        return

    obs_array_list, gf_list, action_list, step_list = [], [], [], []

    Global.clear()
    Global.VERBOSITY = 1

    game_params = episode_data["params"]
    Global.MAX_UNITS = game_params["max_units"]
    Global.UNIT_MOVE_COST = game_params["unit_move_cost"]
    Global.UNIT_SAP_COST = game_params["unit_sap_cost"]
    Global.UNIT_SAP_RANGE = game_params["unit_sap_range"]
    Global.UNIT_SENSOR_RANGE = game_params["unit_sensor_range"]

    state = State(team_id)
    previous_step_unit_array = np.zeros((4, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
    previous_step_sap_array = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float16)
    previous_step_sap_positions = set()
    previous_step_opp_ships = set()
    for episode_observations, episode_actions in zip(
        episode_data["observations"], episode_data["actions"]
    ):
        team_observation = convert_episode_step(episode_observations, team_id)
        team_actions = episode_actions[f"player_{team_id}"]

        state.update(team_observation)

        # print(f"start step {state.global_step}")

        update_exploration_flags(
            state,
            team_actions,
            game_params,
            previous_step_sap_positions,
            previous_step_opp_ships,
        )

        if state.match_step == 0:
            previous_step_unit_array[:] = 0
            previous_step_sap_array[:] = 0
            continue

        if any(
            num_wins > Global.NUM_MATCHES_IN_GAME / 2
            for num_wins in team_observation["team_wins"]
        ):
            break

        is_win = wins[state.match_number]
        if not is_win:
            continue

        obs_array, position_to_action = pars_obs(state, team_actions)

        obs_array[4:8] = previous_step_unit_array
        previous_step_unit_array = obs_array[:4].copy()

        obs_array[8] = previous_step_sap_array
        unit_sap_dropoff_factor = (
            game_params["unit_sap_dropoff_factor"]
            if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND
            else 0.5
        )
        fill_sap_array(
            state, team_actions, previous_step_sap_array, unit_sap_dropoff_factor
        )

        add_to_dataset = True
        if not position_to_action:
            add_to_dataset = False
        if (
            all(x == ActionType.center for x in position_to_action.values())
            and random.random() > 0.1
        ):
            add_to_dataset = False

        if add_to_dataset:

            if team_id == 1:
                obs_array = transpose(obs_array, reflective=True)
                flipped_actions = {}
                for (x, y), action_id in position_to_action.items():
                    x, y = get_opposite(x, y)
                    action_id = ActionType(action_id).transpose(reflective=True).value
                    flipped_actions[(x, y)] = action_id
                position_to_action = flipped_actions

            if (
                Global.OBSTACLE_MOVEMENT_PERIOD_FOUND
                and Global.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            ):
                nebula_tile_drift_direction = (
                    1 if get_nebula_tile_drift_speed() > 0 else -1
                )
                num_steps_before_obstacle_movement = (
                    state.num_steps_before_obstacle_movement()
                )
            else:
                nebula_tile_drift_direction = 0
                num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH

            gf = (
                nebula_tile_drift_direction,
                (
                    game_params["nebula_tile_energy_reduction"] / Global.MAX_UNIT_ENERGY
                    if Global.NEBULA_ENERGY_REDUCTION_FOUND
                    else -1
                ),
                Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY,
                Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY,
                Global.UNIT_SAP_RANGE / Global.SPACE_SIZE,
                (
                    game_params["unit_sap_dropoff_factor"]
                    if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND
                    else -1
                ),
                (
                    game_params["unit_energy_void_factor"]
                    if Global.UNIT_ENERGY_VOID_FACTOR_FOUND
                    else -1
                ),
                state.match_step / Global.MAX_STEPS_IN_MATCH,
                state.match_number / Global.NUM_MATCHES_IN_GAME,
                num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH,
                state.fleet.points / 1000,
                state.opp_fleet.points / 1000,
                state.fleet.reward / 1000,
                state.opp_fleet.reward / 1000,
                sum(Global.RELIC_RESULTS) / 3,
            )

            obs_array_list.append(obs_array)
            gf_list.append(gf)
            action_list.append(
                [[x, y, int(a)] for (x, y), a in position_to_action.items()]
                + [
                    [-1, -1, -1]
                    for _ in range(Global.MAX_UNITS - len(position_to_action))
                ]
            )
            step_list.append(state.global_step)

    return {
        "states": np.array(obs_array_list, dtype=np.float16),
        "gfs": np.array(gf_list, dtype=np.float16),
        "actions": np.array(action_list, dtype=np.int8),
        "steps": np.array(step_list, dtype=np.int16),
    }


def pars_obs(state, team_actions):
    d = np.zeros((17, SPACE_SIZE, SPACE_SIZE), dtype=np.float16)

    # 0 - unit positions
    # 1 - unit energy
    for unit in state.fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[0, y, x] += 1
            d[1, y, x] += unit.energy

    # 2 - opp unit position
    # 3 - opp unit energy
    for unit in state.opp_fleet:
        if unit.energy >= 0:
            x, y = unit.coordinates
            d[2, y, x] += 1
            d[3, y, x] += unit.energy

    d[0] /= 10
    d[1] /= Global.MAX_UNIT_ENERGY
    d[2] /= 10
    d[3] /= Global.MAX_UNIT_ENERGY

    # 4 - previous step unit positions
    # 5 - previous step unit energy
    # 6 - previous step opp unit positions
    # 7 - previous step opp unit energy

    # 8 - previous step sap positions

    f = state.field
    d[9] = f.vision
    d[10] = f.energy / Global.MAX_UNIT_ENERGY
    d[11] = f.asteroid
    d[12] = f.nebulae
    d[13] = f.relic
    d[14] = f.reward
    d[15] = (state.global_step - f.last_relic_check) / Global.MAX_STEPS_IN_MATCH
    d[16] = (state.global_step - f.last_step_in_vision) / Global.MAX_STEPS_IN_MATCH

    actions = {}
    for ship, action in zip(state.fleet.ships, team_actions):
        if ship.node is not None and ship.energy >= 0:
            action_type, dx, dy = action
            position = ship.coordinates
            if position not in actions:
                actions[position] = action_type
            else:
                if action_type != ActionType.center:
                    actions[position] = action_type

    return d, actions


def fill_sap_array(
    state, team_actions, previous_step_sap_array, unit_sap_dropoff_factor
):
    previous_step_sap_array[:] = 0
    for ship, (action_type, dx, dy) in zip(state.fleet.ships, team_actions):
        if action_type == ActionType.sap and ship.node is not None and ship.can_sap():
            sap_x = ship.node.x + dx
            sap_y = ship.node.y + dy
            for x, y in nearby_positions(sap_x, sap_y, 1):
                if x == sap_x and y == sap_y:
                    previous_step_sap_array[y, x] += 1
                else:
                    previous_step_sap_array[y, x] += unit_sap_dropoff_factor
    previous_step_sap_array *= Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY


def apply_actions(state, team_action):
    position_to_action = {}
    for ship, (action_type, dx, dy) in zip(state.fleet.ships, team_action):
        if ship.node is not None and ship.energy >= 0:
            action_type = ActionType(action_type)
            if action_type == ActionType.sap:
                ship.action_queue = [Action(action_type, dx, dy)]
            else:
                ship.action_queue = [Action(action_type)]

            position = ship.coordinates
            if position not in position_to_action:
                position_to_action[position] = action_type
            else:
                if action_type != ActionType.center:
                    position_to_action[position] = action_type

    if state.team_id == 1:
        flipped_actions = {}
        for (x, y), action_id in position_to_action.items():
            x, y = get_opposite(x, y)
            action_id = ActionType(action_id).transpose(reflective=True).value
            flipped_actions[(x, y)] = action_id
        position_to_action = flipped_actions

    return position_to_action


def convert_episodes(submission_id, num_episodes=None, min_opp_score=None):

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    games = pd.read_csv(
        GAMES_PATH, usecols=["SubmissionId", "EpisodeId", "OppSubmissionId"]
    )
    games = games[games["SubmissionId"] == submission_id]

    if min_opp_score is not None:
        submissions_df = pd.read_csv(
            SUBMISSIONS_PATH, usecols=["submission_id", "score"]
        )
        sid_to_score = dict(
            zip(submissions_df["submission_id"], submissions_df["score"])
        )
        games["opp_score"] = [sid_to_score[x] for x in games["OppSubmissionId"]]
        games = games[games["opp_score"] >= min_opp_score]

    episodes = sorted([int(x) for x in games["EpisodeId"].unique()])

    episodes_to_convert = []
    for episode_id in episodes:
        episode_path = f"{EPISODES_DIR}/{episode_id}.json"
        agent_episode_path = f"{OUTPUT_DIR}/{submission_id}_{episode_id}.npz"

        if not os.path.exists(episode_path) or os.path.exists(agent_episode_path):
            continue

        episodes_to_convert.append(episode_id)

    if num_episodes is not None:
        episodes_to_convert = episodes_to_convert[:num_episodes]

    for i, episode_id in enumerate(tqdm(episodes_to_convert), start=1):
        print(f"converting {episode_id}: {i}/{len(episodes_to_convert)}")

        episode_path = f"{EPISODES_DIR}/{episode_id}.json"
        agent_episode_path = f"{OUTPUT_DIR}/{submission_id}_{episode_id}.npz"

        episode_data = json.load(open(episode_path, "r"))
        agents = episode_data["metadata"]["agents"]

        for team_id, agent in enumerate(agents):
            if agent["submission_id"] == submission_id:
                agent_episode = convert_episode(episode_data, team_id)
                if not agent_episode or len(agent_episode["steps"]) == 0:
                    continue
                np.savez(agent_episode_path, **agent_episode)


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


def update_exploration_flags(
    state, actions, game_params, previous_step_sap_positions, previous_step_opp_ships
):

    if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
        # we assume that we have found a constant when the first ship enters Nebula
        for ship in state.fleet:
            if ship.node.type == NodeType.nebula:
                Global.NEBULA_ENERGY_REDUCTION_FOUND = True
                Global.NEBULA_ENERGY_REDUCTION = game_params[
                    "nebula_tile_energy_reduction"
                ]
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
                    break

        previous_step_sap_positions.clear()
        for ship, (action_type, dx, dy) in zip(state.fleet.ships, actions):
            if (
                action_type == ActionType.sap
                and ship.node is not None
                and ship.can_sap()
            ):
                previous_step_sap_positions.add((ship.node.x + dx, ship.node.y + dy))

        previous_step_opp_ships.clear()
        for opp_ship in state.opp_fleet:
            if opp_ship.node is not None and opp_ship.energy > 0:
                previous_step_opp_ships.add(opp_ship.unit_id)

    if not Global.UNIT_ENERGY_VOID_FACTOR_FOUND:
        # we assume that we have found a constant when our ships are next to an enemy ship for the first time.
        for opp_ship in state.opp_fleet:
            for our_ship in state.fleet:
                distance = manhattan_distance(
                    opp_ship.coordinates, our_ship.coordinates
                )
                if distance == 1:
                    Global.UNIT_ENERGY_VOID_FACTOR_FOUND = True
                    Global.UNIT_ENERGY_VOID_FACTOR = game_params[
                        "unit_energy_void_factor"
                    ]
                    break


@dataclass
class Args:
    # submission to convert
    submission_id: int

    # number of episodes to convert
    num_episodes: Optional[int] = None

    # minimum score for an opponent
    min_score: Optional[int] = None


if __name__ == "__main__":
    args = tyro.cli(Args)
    convert_episodes(args.submission_id, args.num_episodes, args.min_score)
