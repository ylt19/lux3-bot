import numpy as np
from sys import stderr as err

from .base import Params
from .path import Action, apply_action
from .space import Node, Space


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0
        self.ships = [Ship(unit_id) for unit_id in range(Params.MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def update(self, obs, space: Space):
        self.points = obs["team_points"][self.team_id]

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            if active:
                ship.node = space.get_node(*position)
                ship.energy = int(energy)
            else:
                ship.clear()

            ship.action_queue = []

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clear()

    def expected_sensor_mask(self):
        space_size = Params.SPACE_SIZE
        sensor_range = Params.UNIT_SENSOR_RANGE
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

        self.task = None
        self.action_queue: list[Action] = []

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None

    def clear(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.action_queue = []

    def can_move(self) -> bool:
        return self.energy >= Params.UNIT_MOVE_COST

    def can_sap(self) -> bool:
        return self.energy >= Params.UNIT_SAP_COST

    def next_position(self) -> tuple[int, int]:
        if not self.can_move() or not self.action_queue:
            return self.coordinates
        return apply_action(*self.coordinates, action=self.action_queue[0].type)
