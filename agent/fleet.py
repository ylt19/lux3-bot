import numpy as np
from collections import defaultdict

from .base import (
    log,
    Global,
    SPACE_SIZE,
    is_inside,
    nearby_positions,
    cardinal_positions,
)
from .path import Action, ActionType, apply_action, actions_to_path
from .space import Node, Space, NodeType


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0
        self.ships = [Ship(unit_id) for unit_id in range(Global.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.is_visible:
                yield ship

    def update(self, obs, space: Space):
        self.points = int(obs["team_points"][self.team_id])

        for ship, visible, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if visible:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
                ship.steps_since_last_viewed = 0
            else:
                if (
                    ship.node is not None
                    and ship.energy >= 0
                    and space.get_node(*ship.coordinates).is_visible
                ):
                    # The ship is out of sight of our sensors
                    ship.steps_since_last_viewed += 1
                    ship.task = None
                    ship.action_queue = []
                else:
                    ship.clear()

            ship.action_queue = []

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clear()

    def expected_sensor_mask(self):
        space_size = Global.SPACE_SIZE
        sensor_range = Global.UNIT_SENSOR_RANGE
        mask = np.zeros((space_size, space_size), dtype=np.int16)
        for ship in self:
            x, y = ship.coordinates
            for _y in range(
                max(0, y - sensor_range), min(space_size, y + sensor_range + 1)
            ):
                mask[_y][
                    max(0, x - sensor_range) : min(space_size, x + sensor_range + 1)
                ] = 1
        return mask


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None
        self.steps_since_last_viewed: int = 0

        self.task = None
        self.action_queue: list[Action] = []

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def is_visible(self) -> True:
        return self.node is not None and self.steps_since_last_viewed == 0

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def path(self):
        if not self.action_queue:
            return [self.coordinates]
        return actions_to_path(self.coordinates, self.action_queue)

    def clear(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.action_queue = []
        self.steps_since_last_viewed = 0

    def can_move(self) -> bool:
        return self.node is not None and self.energy >= Global.UNIT_MOVE_COST

    def can_sap(self) -> bool:
        return self.node is not None and self.energy >= Global.UNIT_SAP_COST

    def next_position(self) -> tuple[int, int]:
        if not self.can_move() or not self.action_queue:
            return self.coordinates
        return apply_action(*self.coordinates, action=self.action_queue[0].type)


def find_hidden_constants(previous_state, state):
    """
    Attempts to discover hidden constants by observing interactions
    between ships and nebulae (NEBULA_ENERGY_REDUCTION) and
    between ships and opponent's ships (UNIT_SAP_DROPOFF_FACTOR, UNIT_ENERGY_VOID_FACTOR).
    """
    _find_nebula_energy_reduction(previous_state, state)
    _find_ship_interaction_constants(previous_state, state)


def _find_nebula_energy_reduction(previous_state, state):
    if Global.NEBULA_ENERGY_REDUCTION_FOUND:
        return

    for previous_ship, ship in zip(previous_state.fleet.ships, state.fleet.ships):
        if not previous_ship.is_visible or not ship.is_visible:
            continue

        node = ship.node
        if node.energy is None:
            continue

        move_cost = 0
        if node != previous_ship.node:
            move_cost = Global.UNIT_MOVE_COST

        if previous_ship.energy < 30 - move_cost:
            continue

        if (
            node.type != NodeType.nebula
            or previous_state.space.get_node(*node.coordinates).type != NodeType.nebula
        ):
            continue

        delta = previous_ship.energy - ship.energy + node.energy - move_cost

        if abs(delta - 25) < 5:
            Global.NEBULA_ENERGY_REDUCTION = 25
        elif abs(delta - 10) < 5:
            Global.NEBULA_ENERGY_REDUCTION = 10
        elif abs(delta - 0) < 5:
            Global.NEBULA_ENERGY_REDUCTION = 0
        else:
            log(
                f"Can't find NEBULA_ENERGY_REDUCTION with ship = {ship}, "
                f"delta = {delta}, step = {state.global_step}",
                level=1,
            )
            continue

        Global.NEBULA_ENERGY_REDUCTION_FOUND = True

        log(
            f"Find param NEBULA_ENERGY_REDUCTION = {Global.NEBULA_ENERGY_REDUCTION}",
            level=2,
        )
        return


def _find_ship_interaction_constants(previous_state, state):
    if Global.UNIT_SAP_DROPOFF_FACTOR_FOUND and Global.UNIT_ENERGY_VOID_FACTOR_FOUND:
        return

    sap_coordinates = []
    for previous_ship in previous_state.fleet:
        action = None
        if previous_ship.action_queue:
            action = previous_ship.action_queue[0]

        if (
            action is None
            or action.type != ActionType.sap
            or not previous_ship.can_sap()
        ):
            continue

        x, y = previous_ship.coordinates
        dx, dy = action.dx, action.dy

        sap_coordinates.append((x + dx, y + dy))

    if not sap_coordinates:
        return

    void_field = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int16)
    for previous_ship, ship in zip(previous_state.fleet.ships, state.fleet.ships):
        if not previous_ship.is_visible or previous_ship.energy <= 0:
            continue

        node = ship.node
        move_cost = 0
        if node is None:
            node = previous_ship.node
        elif node != previous_ship.node:
            move_cost = Global.UNIT_MOVE_COST

        for x_, y_ in cardinal_positions(*node.coordinates):
            void_field[x_, y_] += previous_ship.energy - move_cost

    direct_sap_hits = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int16)
    adjacent_sap_hits = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int16)
    for x, y in sap_coordinates:
        direct_sap_hits[x, y] += 1
        for x_, y_ in nearby_positions(x, y, distance=1):
            adjacent_sap_hits[x_, y_] += 1

    if not Global.UNIT_SAP_DROPOFF_FACTOR_FOUND:
        _find_unit_sap_dropoff_factor(
            previous_state, state, void_field, direct_sap_hits, adjacent_sap_hits
        )

    if not Global.UNIT_ENERGY_VOID_FACTOR_FOUND:
        _find_unit_energy_void_factor(
            previous_state, state, void_field, direct_sap_hits, adjacent_sap_hits
        )


def _find_unit_sap_dropoff_factor(
    previous_state, state, void_field, direct_sap_hits, adjacent_sap_hits
):
    for previous_opp_ship, opp_ship in zip(
        previous_state.opp_fleet.ships, state.opp_fleet.ships
    ):
        if not previous_opp_ship.is_visible or not opp_ship.is_visible:
            continue

        if opp_ship.energy <= 0:
            continue

        x, y = opp_ship.coordinates
        if (
            void_field[x, y] == 0
            and direct_sap_hits[x, y] == 0
            and adjacent_sap_hits[x, y] > 0
        ):
            num_hits = int(adjacent_sap_hits[x, y])

            node = opp_ship.node
            if node.energy is None:
                continue

            move_cost = 0
            if node != previous_opp_ship.node:
                move_cost = Global.UNIT_MOVE_COST

            if previous_state.space.get_node(*node.coordinates).type == NodeType.nebula:
                if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
                    continue
                move_cost += Global.NEBULA_ENERGY_REDUCTION

            delta = previous_opp_ship.energy - opp_ship.energy + node.energy - move_cost

            delta_per_hit = delta / num_hits

            sap_cost = Global.UNIT_SAP_COST

            if abs(delta_per_hit - sap_cost) <= 1:
                Global.UNIT_SAP_DROPOFF_FACTOR = 1
            elif abs(delta_per_hit - sap_cost * 0.5) <= 1:
                Global.UNIT_SAP_DROPOFF_FACTOR = 0.5
            elif abs(delta_per_hit - sap_cost * 0.25) <= 1:
                Global.UNIT_SAP_DROPOFF_FACTOR = 0.25
            else:
                log(
                    f"Can't find UNIT_SAP_DROPOFF_FACTOR with ship = {opp_ship}, "
                    f"delta = {delta}, num_hits = {num_hits}, step = {state.global_step}",
                    level=1,
                )
                continue

            Global.UNIT_SAP_DROPOFF_FACTOR_FOUND = True

            log(
                f"Find param UNIT_SAP_DROPOFF_FACTOR = {Global.UNIT_SAP_DROPOFF_FACTOR}",
                level=2,
            )
            return


def _find_unit_energy_void_factor(
    previous_state, state, void_field, direct_sap_hits, adjacent_sap_hits
):
    position_to_unit_count = defaultdict(int)
    for opp_ship in state.opp_fleet:
        if opp_ship.energy >= 0:
            position_to_unit_count[opp_ship.coordinates] += 1

    for previous_opp_ship, opp_ship in zip(
        previous_state.opp_fleet.ships, state.opp_fleet.ships
    ):
        if not previous_opp_ship.is_visible or not opp_ship.is_visible:
            continue

        if opp_ship.energy <= 0:
            continue

        x, y = opp_ship.coordinates
        if (
            void_field[x, y] > 0
            and direct_sap_hits[x, y] == 0
            and adjacent_sap_hits[x, y] == 0
        ):
            node_void_field = int(void_field[x, y])
            node_unit_count = position_to_unit_count[(x, y)]

            node = opp_ship.node
            if node.energy is None:
                continue

            move_cost = 0
            if node != previous_opp_ship.node:
                move_cost = Global.UNIT_MOVE_COST

            if previous_state.space.get_node(*node.coordinates).type == NodeType.nebula:
                if not Global.NEBULA_ENERGY_REDUCTION_FOUND:
                    continue
                move_cost += Global.NEBULA_ENERGY_REDUCTION

            delta = previous_opp_ship.energy - opp_ship.energy + node.energy - move_cost

            options = [0.0625, 0.125, 0.25, 0.375]
            results = []
            for option in options:
                expected = node_void_field / node_unit_count * option
                result = abs(expected - delta) <= 1
                results.append(result)

            if sum(results) == 1:
                for option, result in zip(options, results):
                    if result:
                        Global.UNIT_ENERGY_VOID_FACTOR = option

                Global.UNIT_ENERGY_VOID_FACTOR_FOUND = True

                log(
                    f"Find param UNIT_ENERGY_VOID_FACTOR = {Global.UNIT_ENERGY_VOID_FACTOR}",
                    level=2,
                )
                return

            if sum(results) == 0:
                log(
                    f"Can't find UNIT_ENERGY_VOID_FACTOR with ship = {opp_ship}, "
                    f"delta = {delta}, void_field = {node_void_field}, step = {state.global_step}",
                    level=1,
                )
