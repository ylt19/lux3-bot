import numpy as np
from scipy.signal import convolve2d

from .base import log, Global, SPACE_SIZE
from .path import Action, ActionType
from .state import State
from .tasks import HarvestTask, GatherEnergy


def sap(state: State):
    sap_range = Global.UNIT_SAP_RANGE
    sap_diameter = sap_range * 2 + 1
    opp_ships_array = np.zeros(
        (SPACE_SIZE + 2 * sap_range, SPACE_SIZE + 2 * sap_range), dtype=np.float32
    )

    for opp_ship in state.opp_fleet:
        if opp_ship.energy >= 0:
            x, y = opp_ship.coordinates
            opp_ships_array[y + sap_range][x + sap_range] += 1

    for ship in state.fleet:
        if not ship.can_sap():
            continue

        if not isinstance(ship.task, (type(None), HarvestTask, GatherEnergy)):
            continue

        x, y = ship.coordinates
        local_opp_ships_array = opp_ships_array[
            y : y + sap_diameter, x : x + sap_diameter
        ]

        if not local_opp_ships_array.any():
            continue

        local_opp_ships_array = convolve2d(
            local_opp_ships_array,
            get_sap_kernel(),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        max_score = local_opp_ships_array.max()
        if ship.energy > 350:
            if max_score < 1.0:
                continue
        else:
            if max_score < 1.5:
                continue

        sap_direction = np.where(local_opp_ships_array == max_score)
        dx = sap_direction[1][0] - sap_range
        dy = sap_direction[0][0] - sap_range

        # log(ship, max_score, dx, dy)
        # log(local_opp_ships_array)

        ship.action_queue = [Action(ActionType.sap, dx, dy)]


def get_sap_kernel():
    a = np.zeros((3, 3), dtype=np.float32)
    a[:] = Global.UNIT_SAP_DROPOFF_FACTOR
    a[1, 1] = 1
    return a
