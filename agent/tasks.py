import torch
import numpy as np
from scipy.signal import convolve2d

from .base import (
    log,
    Global,
    SPACE_SIZE,
    transpose,
    get_opposite,
    nearby_positions,
    get_nebula_tile_drift_speed,
)
from .path import Action, ActionType
from .exploration import VoidSeeker, RelicFinder
from .heal import Heal
from .sap import sap
from .msg import print_msg
from .control import Control


def find_moves(agent):
    if Global.Params.IL:
        apply_nn(agent.state, agent.previous_state, agent.unit_model, agent.sap_model)
    else:
        apply_rb(agent.state)


def apply_rb(state):

    for ship in state.fleet:
        task = ship.task
        if task is None:
            continue

        if task.completed(state, ship):
            ship.task = None
            continue

        if task.apply(state, ship):
            ship.task = task
        else:
            ship.task = None

    tasks = generate_tasks(state)

    scores = []
    for ship in state.fleet:
        if ship.task is None:
            for task in tasks:
                if task.ship and task.ship != ship:
                    continue
                score = task.evaluate(state, ship)
                if score <= 0:
                    continue
                scores.append({"ship": ship, "task": task, "score": score})

    scores = sorted(scores, key=lambda x: -x["score"])

    tasks_closed = set()
    ships_closed = set()

    for d in scores:
        ship = d["ship"]
        task = d["task"]

        if ship in ships_closed or task in tasks_closed:
            continue

        if task.apply(state, ship):
            ship.task = task

            ships_closed.add(ship)
            tasks_closed.add(task)

    if Global.Params.MSG_TASK:
        print_msg(state)
        return

    # evasion(state)
    sap(state)


def apply_nn(state, previous_state, unit_model, sap_model):
    with torch.no_grad():
        obs, gf = create_unit_nn_input(state, previous_state)
        p = unit_model(
            torch.from_numpy(obs).unsqueeze(0), torch.from_numpy(gf).unsqueeze(0)
        )
    policy = p.squeeze(0).numpy()

    nodes_with_action = set()
    for ship in state.fleet:

        if ship.node in nodes_with_action:
            continue

        nodes_with_action.add(ship.node)

        if state.team_id == 0:
            x, y = ship.coordinates
            label = np.argsort(policy[:, y, x])[::-1]
            label = int(label[0])
        else:
            x, y = get_opposite(*ship.coordinates)
            label = np.argsort(policy[:, y, x])[::-1]
            label = int(label[0])
            label = ActionType(label).transpose(reflective=True).value

        if label == ActionType.sap:
            obs, gf = create_sap_nn_input(state, previous_state, ship)
            with torch.no_grad():
                p = sap_model(
                    torch.from_numpy(obs).unsqueeze(0),
                    torch.from_numpy(gf).unsqueeze(0),
                )
                p = torch.sigmoid(p)
                sap_policy = p.squeeze(0).squeeze(0).numpy()
                if state.team_id == 1:
                    sap_policy = transpose(sap_policy, reflective=True)

            r = Global.UNIT_SAP_RANGE * 2 + 1
            sap_kernel = np.ones((r, r), dtype=np.float32)

            ship_sap_field = np.zeros((SPACE_SIZE, SPACE_SIZE), np.float32)
            ship_sap_field[ship.node.y, ship.node.x] = 1
            ship_sap_field = convolve2d(
                ship_sap_field,
                sap_kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )

            sap_policy = sap_policy * ship_sap_field

            max_policy = sap_policy.max()
            # log(f"{ship} sap with prob {max_policy}")
            if max_policy > 0.5:
                sap_y, sap_x = np.where(sap_policy == max_policy)
                sap_y, sap_x = int(sap_y[0]), int(sap_x[0])
                if state.field.sap_mask[sap_y, sap_x] == 0:
                    log(
                        f"Ignore a pointless sap action: {ship} -> {sap_x, sap_y}, step {state.global_step}",
                        level=2,
                    )
                    ship.action_queue = [Action(ActionType.center)]
                else:
                    dx = sap_x - ship.node.x
                    dy = sap_y - ship.node.y
                    ship.action_queue = [Action(ActionType.sap, dx, dy)]
            else:
                ship.action_queue = [Action(ActionType.center)]

        else:
            ship.action_queue = [Action(ActionType(label))]


def generate_tasks(state):
    tasks = []

    p = Global.Params

    if p.RELIC_FINDER_TASK:
        tasks += RelicFinder.generate_tasks(state)

    if p.VOID_SEEKER_TASK:
        tasks += VoidSeeker.generate_tasks(state)

    if p.CONTROL_TASK:
        tasks += Control.generate_tasks(state)

    if p.HEAL_TASK:
        tasks += Heal.generate_tasks(state)

    return tasks


def get_sap_array(previous_state):
    sap_array = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    for unit in previous_state.fleet:
        if (
            unit.action_queue
            and unit.action_queue[0].type == ActionType.sap
            and unit.node is not None
            and unit.can_sap()
        ):
            action = unit.action_queue[0]
            sap_x = unit.node.x + action.dx
            sap_y = unit.node.y + action.dy
            for x, y in nearby_positions(sap_x, sap_y, 1):
                if x == sap_x and y == sap_y:
                    sap_array[y, x] += 1
                else:
                    sap_array[y, x] += Global.UNIT_SAP_DROPOFF_FACTOR_FOUND

    sap_array *= Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY

    # if sap_array.any():
    #     show_field(sap_array * 400)

    return sap_array


def create_unit_nn_input(state, previous_state):

    gf = np.zeros((14, 3, 3), dtype=np.float32)

    if Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
        nebula_tile_drift_direction = 1 if get_nebula_tile_drift_speed() > 0 else -1
        num_steps_before_obstacle_movement = state.num_steps_before_obstacle_movement()
    else:
        nebula_tile_drift_direction = 0
        num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH

    gf[0] = nebula_tile_drift_direction
    gf[1] = (
        Global.NEBULA_ENERGY_REDUCTION / Global.MAX_UNIT_ENERGY
        if Global.NEBULA_ENERGY_REDUCTION_FOUND
        else -1
    )
    gf[2] = Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY
    gf[3] = Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY
    gf[4] = Global.UNIT_SAP_RANGE / Global.SPACE_SIZE
    gf[5] = (
        Global.UNIT_SAP_DROPOFF_FACTOR if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND else -1
    )
    gf[6] = (
        Global.UNIT_ENERGY_VOID_FACTOR if Global.UNIT_ENERGY_VOID_FACTOR_FOUND else -1
    )
    gf[7] = state.match_step / Global.MAX_STEPS_IN_MATCH
    gf[8] = state.match_number / Global.NUM_MATCHES_IN_GAME
    gf[9] = num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH
    gf[10] = state.fleet.points / 1000
    gf[11] = state.opp_fleet.points / 1000
    gf[12] = state.fleet.reward / 1000
    gf[13] = state.opp_fleet.reward / 1000

    d = np.zeros((15, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)

    for unit in state.fleet:
        x, y = unit.coordinates
        d[0, y, x] += 1 / 10
        d[1, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    for unit in state.opp_fleet:
        x, y = unit.coordinates
        d[2, y, x] += 1 / 10
        d[3, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    for unit in previous_state.fleet:
        x, y = unit.coordinates
        d[4, y, x] += 1 / 10
        d[5, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    for unit in previous_state.opp_fleet:
        x, y = unit.coordinates
        d[6, y, x] += 1 / 10
        d[7, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    d[8] = get_sap_array(previous_state)

    d[9] = state.field.vision
    d[10] = state.field.energy / Global.MAX_UNIT_ENERGY
    d[11] = state.field.asteroid
    d[12] = state.field.nebulae
    d[13] = state.field.relic
    d[14] = state.field.reward
    # d[15] = state.field.unexplored_for_reward

    if state.team_id == 1:
        d = transpose(d, reflective=True).copy()

    return d, gf


def create_sap_nn_input(state, previous_state, ship):

    gf = np.zeros((15, 3, 3), dtype=np.float32)

    if Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
        nebula_tile_drift_direction = 1 if get_nebula_tile_drift_speed() > 0 else -1
        num_steps_before_obstacle_movement = state.num_steps_before_obstacle_movement()
    else:
        nebula_tile_drift_direction = 0
        num_steps_before_obstacle_movement = -Global.MAX_STEPS_IN_MATCH

    gf[0] = nebula_tile_drift_direction
    gf[1] = (
        Global.NEBULA_ENERGY_REDUCTION / Global.MAX_UNIT_ENERGY
        if Global.NEBULA_ENERGY_REDUCTION_FOUND
        else -1
    )
    gf[2] = Global.UNIT_MOVE_COST / Global.MAX_UNIT_ENERGY
    gf[3] = Global.UNIT_SAP_COST / Global.MAX_UNIT_ENERGY
    gf[4] = Global.UNIT_SAP_RANGE / Global.SPACE_SIZE
    gf[5] = (
        Global.UNIT_SAP_DROPOFF_FACTOR if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND else -1
    )
    gf[6] = (
        Global.UNIT_ENERGY_VOID_FACTOR if Global.UNIT_ENERGY_VOID_FACTOR_FOUND else -1
    )
    gf[7] = state.match_step / Global.MAX_STEPS_IN_MATCH
    gf[8] = state.match_number / Global.NUM_MATCHES_IN_GAME
    gf[9] = num_steps_before_obstacle_movement / Global.MAX_STEPS_IN_MATCH
    gf[10] = state.fleet.points / 1000
    gf[11] = state.opp_fleet.points / 1000
    gf[12] = state.fleet.reward / 1000
    gf[13] = state.opp_fleet.reward / 1000
    gf[14] = ship.energy / Global.MAX_UNIT_ENERGY

    d = np.zeros((16, SPACE_SIZE, SPACE_SIZE), dtype=np.float32)

    for unit in state.fleet:
        x, y = unit.coordinates
        d[0, y, x] += 1 / 10
        d[1, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    for unit in state.opp_fleet:
        x, y = unit.coordinates
        d[2, y, x] += 1 / 10
        d[3, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    for unit in previous_state.fleet:
        x, y = unit.coordinates
        d[4, y, x] += 1 / 10
        d[5, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    for unit in previous_state.opp_fleet:
        x, y = unit.coordinates
        d[6, y, x] += 1 / 10
        d[7, y, x] += unit.energy / Global.MAX_UNIT_ENERGY

    d[8] = get_sap_array(previous_state)

    d[9] = state.field.vision
    d[10] = state.field.energy / Global.MAX_UNIT_ENERGY
    d[11] = state.field.asteroid
    d[12] = state.field.nebulae
    d[13] = state.field.relic
    d[14] = state.field.reward
    # d[15] = state.field.unexplored_for_reward

    ship_arr = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int32)
    ship_arr[ship.node.y, ship.node.x] = 1

    r = Global.UNIT_SAP_RANGE * 2 + 1
    sap_kernel = np.ones((r, r), dtype=np.int32)

    ship_arr = convolve2d(
        ship_arr,
        sap_kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )

    d[15] = ship_arr

    if state.team_id == 1:
        d = transpose(d, reflective=True).copy()

    return d, gf
