import math
from collections import defaultdict

from .base import (
    log,
    Global,
    chebyshev_distance,
    nearby_positions,
)
from .path import Action, ActionType
from .state import State

SAP_OOV_DAMAGE_TH = 50


def get_sap_damage_th(ship_energy):
    if ship_energy < 0.75 * Global.MAX_UNIT_ENERGY:
        return 0.9 * Global.UNIT_SAP_COST

    if ship_energy < 0.85 * Global.MAX_UNIT_ENERGY:
        return 0.5 * Global.UNIT_SAP_COST

    return 0.2 * Global.UNIT_SAP_COST


class SapInfo:
    def __init__(self, target):
        self.target = target

        self.opp_ships = []

        self.oov_direct_hit_prob = 0
        self.oov_adjacent_targets = set()
        self.oov_adjacent_hit_probs = []

    def __repr__(self):
        return (
            f"SapInfo(target={self.target.coordinates}, opp_ships={self.opp_ships}, "
            f"oov=({self.oov_direct_hit_prob:.2f}, {self.mean_oov_adjacent_hit():.2f}))"
        )

    def mean_oov_adjacent_hit(self):
        return sum(i * p for i, p in enumerate(self.oov_adjacent_hit_probs))

    def estimate_damage(self, node_to_sap_damage):
        sap_damage = 0
        for opp_ship in self.opp_ships:
            node_damage = node_to_sap_damage.get(opp_ship.node, 0)

            if opp_ship.energy - node_damage >= 0:
                if opp_ship.node == self.target:
                    if opp_ship.can_move():
                        sap_damage += (
                            Global.UNIT_SAP_COST * Global.UNIT_SAP_DROPOFF_FACTOR
                        )
                    else:
                        sap_damage += Global.UNIT_SAP_COST
                else:
                    if opp_ship.can_move():
                        sap_damage += (
                            Global.UNIT_SAP_COST * Global.UNIT_SAP_DROPOFF_FACTOR * 0.3
                        )
                    else:
                        sap_damage += (
                            Global.UNIT_SAP_COST * Global.UNIT_SAP_DROPOFF_FACTOR
                        )

        target_damage = node_to_sap_damage.get(self.target, 0)
        if SAP_OOV_DAMAGE_TH - target_damage >= 0:
            sap_damage += self.oov_direct_hit_prob * Global.UNIT_SAP_COST

        for node in self.oov_adjacent_targets:
            node_damage = node_to_sap_damage.get(node, 0)

            if SAP_OOV_DAMAGE_TH - node_damage >= 0:
                sap_damage += (
                    self.mean_oov_adjacent_hit()
                    * Global.UNIT_SAP_COST
                    * Global.UNIT_SAP_DROPOFF_FACTOR
                    / len(self.oov_adjacent_targets)
                )

        return sap_damage


def binomial_coefficient(x, y):
    return math.factorial(x) / (math.factorial(y) * math.factorial(x - y))


def hypergeometric_distribution(n, b, s, x):
    """
    Calculate the probability of finding exactly 'x' balls in 's' randomly selected baskets.

    Parameters:
    - n: Total number of baskets.
    - b: Total number of balls (each basket can contain at most 1 ball).
    - s: Number of baskets randomly selected.
    - x: The exact number of balls you want to find in the selected baskets.

    Returns:
    - float: The probability of finding exactly 'x' balls in the 's' selected baskets.
    """
    if b < x or n < s or n < b or s < x or n - s < b - x:
        return 0.0

    return (
        binomial_coefficient(s, x)
        * binomial_coefficient(n - s, b - x)
        / binomial_coefficient(n, b)
    )


def prob_opp_on_rewards(state):
    if not Global.ALL_REWARDS_FOUND:
        return 0, set()

    # invisible rewards that can be reachable by the opponent
    reward_nodes = set()
    for reward_node in state.space.reward_nodes:
        if not reward_node.is_visible:
            if (
                state.opp_fleet.spawn_distance(*reward_node.coordinates)
                <= state.match_step
            ):
                reward_nodes.add(reward_node)

    # how many rewards will the opponent score according to our vision
    opp_rewards_in_vision = 0
    for ship in state.opp_fleet:
        if ship.node.reward:
            opp_rewards_in_vision += 1

    num_opp_ships_with_rewards = state.opp_fleet.reward - opp_rewards_in_vision

    return num_opp_ships_with_rewards, reward_nodes


def create_sap_targets(state: State):
    node_to_opp_ships = defaultdict(list)
    for opp_ship in state.opp_fleet:
        if opp_ship.energy >= 0:
            node_to_opp_ships[opp_ship.node].append(opp_ship)

    sap_nodes = set()
    for ship_node in node_to_opp_ships:
        for xy_sap in nearby_positions(*ship_node.coordinates, 1):
            sap_nodes.add(state.space.get_node(*xy_sap))

    num_opp_ships_with_rewards, reward_nodes = prob_opp_on_rewards(state)
    for reward_node in reward_nodes:
        for xy_sap in nearby_positions(*reward_node.coordinates, 1):
            sap_nodes.add(state.space.get_node(*xy_sap))

    sap_targets = []
    for target in sap_nodes:

        sap_info = SapInfo(target)

        local_opp_ships = []
        for xy_sap in nearby_positions(*target.coordinates, 1):
            n = state.space.get_node(*xy_sap)
            for opp_ship in node_to_opp_ships.get(n, []):
                local_opp_ships.append(opp_ship)

        if local_opp_ships:
            sap_info.opp_ships = local_opp_ships

        if num_opp_ships_with_rewards and reward_nodes:
            oov_direct_hit_prob, oov_adjacent_hit_probs, oov_adjacent_targets = (
                calculate_out_of_vision_probs(
                    state, target, num_opp_ships_with_rewards, reward_nodes
                )
            )
            if oov_direct_hit_prob > 0 or (
                oov_adjacent_hit_probs and any(x > 0 for x in oov_adjacent_hit_probs)
            ):
                sap_info.oov_adjacent_targets = set(oov_adjacent_targets)
                sap_info.oov_direct_hit_prob = oov_direct_hit_prob
                sap_info.oov_adjacent_hit_probs = oov_adjacent_hit_probs

        sap_targets.append(sap_info)

    return sap_targets


def calculate_out_of_vision_probs(
    state, target, num_opp_ships_with_rewards, reward_nodes
):
    oov_direct_hit_prob = 0
    if target in reward_nodes:
        oov_direct_hit_prob = num_opp_ships_with_rewards / len(reward_nodes)

    oov_adjacent_targets = []
    num_adjacent_reward_nodes = 0
    for xy_sap in nearby_positions(*target.coordinates, 1):
        n = state.space.get_node(*xy_sap)
        if n in reward_nodes and n != target:
            oov_adjacent_targets.append(n)
            num_adjacent_reward_nodes += 1

    oov_adjacent_hit_probs = []
    for num_ships in range(num_adjacent_reward_nodes + 1):
        p = hypergeometric_distribution(
            len(reward_nodes),
            num_opp_ships_with_rewards,
            num_adjacent_reward_nodes,
            num_ships,
        )
        oov_adjacent_hit_probs.append(p)

    return oov_direct_hit_prob, oov_adjacent_hit_probs, oov_adjacent_targets


def sap(state: State):
    sap_targets = create_sap_targets(state)
    if not sap_targets:
        return

    node_to_sap_damage = defaultdict(int)

    for ship in sorted(state.fleet, key=lambda s: -s.energy):
        if not ship.can_sap():
            continue

        ship_sap_targets = [
            x
            for x in sap_targets
            if chebyshev_distance(ship.coordinates, x.target.coordinates)
            <= Global.UNIT_SAP_RANGE
        ]
        if not ship_sap_targets:
            continue

        ship_sap_targets = sorted(
            ship_sap_targets, key=lambda x: -x.estimate_damage(node_to_sap_damage)
        )

        sap_target = ship_sap_targets[0]

        damage = sap_target.estimate_damage(node_to_sap_damage)

        if damage < get_sap_damage_th(ship.energy):
            continue

        target_node = sap_target.target

        dx = target_node.coordinates[0] - ship.node.x
        dy = target_node.coordinates[1] - ship.node.y

        ship.action_queue = [Action(ActionType.sap, dx, dy)]

        node_to_sap_damage[target_node] += Global.UNIT_SAP_COST
        for xy_sap in nearby_positions(*target_node.coordinates, 1):
            nearby_node = state.space.get_node(*xy_sap)
            if nearby_node != target_node:
                node_to_sap_damage[nearby_node] += (
                    Global.UNIT_SAP_COST * Global.UNIT_SAP_DROPOFF_FACTOR
                )
