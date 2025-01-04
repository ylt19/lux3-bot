from collections import defaultdict

from .base import (
    log,
    Global,
    chebyshev_distance,
    nearby_positions,
    manhattan_distance,
)
from .path import Action, ActionType
from .state import State


class OppShip:
    def __init__(self, node, unit_id=None, energy=50, probability=1):
        self.node = node
        self.unit_id = unit_id
        self.energy = energy
        self.probability = probability

    @classmethod
    def from_ship(cls, ship):
        return OppShip(
            node=ship.node, unit_id=ship.unit_id, energy=ship.energy, probability=1
        )

    def __repr__(self):
        return f"Ship({self.node.coordinates}, {self.energy})"


class SapInfo:
    def __init__(self, target):
        self.target = target
        self.direct_hits: set[OppShip] = set()
        self.nearby_hits: set[OppShip] = set()

    def __repr__(self):
        return (
            f"SapInfo(target={self.target.coordinates}, direct={self.direct_hits}, "
            f"nearby={self.nearby_hits}"
        )

    def empty(self):
        return len(self.direct_hits) == 0 and len(self.nearby_hits) == 0

    def estimate_damage(self):
        damage = 0
        for d in self.direct_hits:
            if d.energy >= 0:
                damage += Global.UNIT_SAP_COST * d.probability
        for d in self.nearby_hits:
            if d.energy >= 0:
                damage += (
                    Global.UNIT_SAP_COST
                    * d.probability
                    * Global.UNIT_SAP_DROPOFF_FACTOR
                    * 0.2
                )
        return damage


def sap(state: State):
    xy_to_opp_ships = find_opp_ships_positions(state)

    for ship in sorted(state.fleet, key=lambda s: -s.energy):
        if not ship.can_sap():
            continue

        xy_to_sap_info = {}
        for xy in nearby_positions(*ship.coordinates, Global.UNIT_SAP_RANGE + 1):

            if xy not in xy_to_opp_ships:
                continue

            direct_hits = [
                s for s in xy_to_opp_ships[xy] if s.energy >= 0 and s.probability > 0
            ]
            if not direct_hits:
                continue

            for xy_sap in nearby_positions(*xy, 1):
                if xy_sap not in xy_to_sap_info:
                    sap_target = state.space.get_node(*xy_sap)
                    xy_to_sap_info[xy_sap] = SapInfo(sap_target)

                sap_info = xy_to_sap_info[xy_sap]
                if xy_sap == xy:
                    for opp_ship in direct_hits:
                        sap_info.direct_hits.add(opp_ship)
                else:
                    for opp_ship in direct_hits:
                        sap_info.nearby_hits.add(opp_ship)

        if not xy_to_sap_info:
            continue

        targets = [
            x
            for x in xy_to_sap_info.values()
            if not x.empty()
            and chebyshev_distance(x.target.coordinates, ship.coordinates)
            <= Global.UNIT_SAP_RANGE
        ]
        targets = sorted(targets, key=lambda x: -x.estimate_damage())
        if not targets:
            continue

        target = targets[0]
        dx = target.target.coordinates[0] - ship.node.x
        dy = target.target.coordinates[1] - ship.node.y

        ship.action_queue = [Action(ActionType.sap, dx, dy)]

        # log(
        #     f"add sap {ship}->{dx, dy}, target={target.direct_hits, target.nearby_hits}"
        # )
        # log(xy_to_opp_ships)

        for opp_ship in target.direct_hits:
            opp_ship.energy -= Global.UNIT_SAP_COST

        for opp_ship in target.nearby_hits:
            opp_ship.energy -= Global.UNIT_SAP_COST * Global.UNIT_SAP_DROPOFF_FACTOR


def find_opp_ships_positions(state):
    xy_to_opp_ships = defaultdict(list)
    for opp_ship in state.opp_fleet:
        if opp_ship.energy >= 0:
            xy_to_opp_ships[opp_ship.coordinates].append(OppShip.from_ship(opp_ship))

    num_opp_ships_with_rewards, reward_nodes = prob_opp_on_rewards(state)
    if reward_nodes and num_opp_ships_with_rewards > 0:
        prob = num_opp_ships_with_rewards / len(reward_nodes)
        for node in reward_nodes:
            opp_ship = OppShip(node=node, probability=prob)
            xy_to_opp_ships[node.coordinates].append(opp_ship)

    return xy_to_opp_ships


def prob_opp_on_rewards(state):
    if not Global.ALL_REWARDS_FOUND:
        return 0, set()

    opp_spawn_position = state.opp_fleet.spawn_position

    # invisible rewards that can be reachable by the opponent
    reward_nodes = set()
    for reward_node in state.space.reward_nodes:
        if not reward_node.is_visible and (
            manhattan_distance(reward_node.coordinates, opp_spawn_position)
            <= state.match_step
        ):
            reward_nodes.add(reward_node)

    # how many rewards will the opponent score according to our vision
    opp_rewards_in_vision = 0
    for ship in state.opp_fleet:
        if ship.node.reward:
            opp_rewards_in_vision = +1

    num_opp_ships_with_rewards = state.opp_fleet.reward - opp_rewards_in_vision

    return num_opp_ships_with_rewards, reward_nodes
