import copy
import numpy as np
from functools import cached_property
from pathfinding import Grid as w9_Grid, ResumableDijkstra, ReservationTable

from .path import NodeType
from .base import Global, warp_point


class Grid:
    def __init__(self, state):
        self._state = state

        self._resumable_search = [
            [None for _ in range(Global.MAX_UNITS)],
            [None for _ in range(Global.MAX_UNITS)],
        ]

    @property
    def space(self):
        return self._state.space

    @cached_property
    def energy(self):
        ground = Global.Params.ENERGY_TO_WEIGHT_GROUND

        weights = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
        for node in self.space:
            energy = node.energy
            if energy is None:
                energy = Global.HIDDEN_NODE_ENERGY
            weights[node.y][node.x] = self.energy_to_weight(energy, ground)

        return w9_Grid(weights, pause_action_cost="node.weight")

    @cached_property
    def energy_with_low_ground(self):
        ground = Global.UNIT_MOVE_COST

        weights = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
        for node in self.space:
            energy = node.energy
            if energy is None:
                energy = Global.HIDDEN_NODE_ENERGY
            weights[node.y][node.x] = self.energy_to_weight(energy, ground)

        return w9_Grid(weights, pause_action_cost="node.weight")

    @cached_property
    def energy_gain(self):
        ground = Global.Params.ENERGY_TO_WEIGHT_GROUND

        weights = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
        for node in self.space:
            weights[node.y][node.x] = self.energy_to_weight(node.energy_gain, ground)

        return w9_Grid(weights, pause_action_cost="node.weight")

    @cached_property
    def energy_gain_with_asteroids(self):
        ground = Global.Params.ENERGY_TO_WEIGHT_GROUND

        weights = np.zeros((Global.SPACE_SIZE, Global.SPACE_SIZE), np.float32)
        for node in self.space:

            if not node.is_walkable:
                w = -1
            else:
                w = self.energy_to_weight(node.energy_gain, ground)

            weights[node.y][node.x] = w

        return w9_Grid(weights, pause_action_cost="node.weight")

    @staticmethod
    def energy_to_weight(energy, ground):
        if energy < ground:
            return ground - energy + 1
        return Global.Params.ENERGY_TO_WEIGHT_BASE ** (ground - energy)

    def resumable_search(self, unit_id, team_id=None):
        if team_id is None:
            team_id = self._state.team_id

        team_resumable_search = self._resumable_search[team_id]
        if team_resumable_search[unit_id] is not None:
            return team_resumable_search[unit_id]

        fleet = self._state.get_fleet(team_id)
        ship = fleet.ships[unit_id]
        if ship.node is None:
            return

        grid = self.energy_gain_with_asteroids
        if grid.has_obstacle(ship.coordinates):
            grid = copy.copy(grid)
            grid.remove_obstacle(ship.coordinates)

        rs = ResumableDijkstra(grid, ship.coordinates)
        team_resumable_search[unit_id] = rs
        return rs

    @cached_property
    def reservation_table(self):
        reservation_table = ReservationTable(self.energy)
        add_dynamic_environment(reservation_table, self._state)
        return reservation_table


def add_dynamic_environment(rt, state):
    shift = Global.OBSTACLE_MOVEMENT_DIRECTION

    for node in state.space:
        if node.type == NodeType.asteroid:
            point = node.coordinates
            path = []
            match_step = state.match_step
            global_step = state.global_step
            while match_step <= Global.MAX_STEPS_IN_MATCH:
                if (
                    len(path) > 0
                    and (global_step - 1) % Global.OBSTACLE_MOVEMENT_PERIOD == 0
                ):
                    rt.add_vertex_constraint(time=len(path), node=point)
                    point = warp_point(point[0] + shift[0], point[1] + shift[1])
                path.append(point)
                match_step += 1
                global_step += 1

            rt.add_path(path, reserve_destination=False)

        elif node.type == NodeType.nebula and Global.NEBULA_ENERGY_REDUCTION != 0:
            point = node.coordinates
            path = []
            match_step = state.match_step
            global_step = state.global_step
            while match_step <= Global.MAX_STEPS_IN_MATCH:
                if (
                    len(path) > 1
                    and (global_step - 2) % Global.OBSTACLE_MOVEMENT_PERIOD == 0
                ):
                    point = warp_point(point[0] + shift[0], point[1] + shift[1])
                path.append(point)
                match_step += 1
                global_step += 1

            rt.add_weight_path(path, weight=Global.NEBULA_ENERGY_REDUCTION)

    return rt
