import os
import sys
import numpy as np

IS_KAGGLE = os.path.exists("/kaggle_simulations")


class Global:
    VERBOSITY = 1
    if IS_KAGGLE:
        VERBOSITY = -1

    # Game related constants
    MAX_UNITS = 16
    SPACE_SIZE = 24
    MAX_UNIT_ENERGY = 400
    RELIC_REWARD_RANGE = 2
    MIN_ENERGY_PER_TILE = -20
    MAX_ENERGY_PER_TILE = 20
    MAX_STEPS_IN_MATCH = 100
    NUM_MATCHES_IN_GAME = 5
    UNIT_MOVE_COST = 1  # OPTIONS: list(range(1, 6))
    UNIT_SAP_COST = 30  # OPTIONS: list(range(30, 51))
    UNIT_SAP_RANGE = 3  # OPTIONS: list(range(3, 8))
    UNIT_SENSOR_RANGE = 2  # OPTIONS: [1, 2, 3, 4]
    NEBULA_ENERGY_REDUCTION = 5  # OPTIONS: [0, 1, 2, 3, 5, 25]
    OBSTACLE_MOVEMENT_PERIOD = 20  # OPTIONS: 6.66, 10, 20, 40
    OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]
    UNIT_SAP_DROPOFF_FACTOR = 0.5  # OPTIONS: [0.25, 0.5, 1]
    UNIT_ENERGY_VOID_FACTOR = 0.125  # OPTIONS: [0.0625, 0.125, 0.25, 0.375]
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 50
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 2

    # Exploration flags
    ALL_RELICS_FOUND = False
    ALL_REWARDS_FOUND = False
    NEBULA_ENERGY_REDUCTION_FOUND = False
    OBSTACLE_MOVEMENT_PERIOD_FOUND = False
    OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
    UNIT_SAP_DROPOFF_FACTOR_FOUND = False
    UNIT_ENERGY_VOID_FACTOR_FOUND = False

    # Info about completed matches
    NUM_COMPLETED_MATCHES = 0
    NUM_WINS = 0
    POINTS = []  # points we scored
    OPP_POINTS = []  # points scored by the opponent

    # Game logs:

    # REWARD_RESULTS: [{"nodes": Set[Node], "points": int}, ...]
    # A history of reward events, where each entry contains:
    # - "nodes": A set of nodes where our ships were located.
    # - "points": The number of points scored at that location.
    # This data will help identify which nodes yield points.
    REWARD_RESULTS = []

    # obstacles_movement_status: list of bool
    # A history log of obstacle (asteroids and nebulae) movement events.
    # - `True`: The ships' sensors detected a change in the obstacles' positions at this step.
    # - `False`: The sensors did not detect any changes.
    # This information will be used to determine the speed and direction of obstacle movement.
    OBSTACLES_MOVEMENT_STATUS = []

    # Game Params:
    class DefaultParams:
        IL = True

        HIDDEN_NODE_ENERGY = 0
        ENERGY_TO_WEIGHT_BASE = 1.2
        ENERGY_TO_WEIGHT_GROUND = 12

        RELIC_FINDER_TASK = True
        RELIC_FINDER_NUM_TASKS = 10
        RELIC_FINDER_INIT_SCORE = 1000
        RELIC_FINDER_PATH_LENGTH_MULTIPLIER = -5
        RELIC_FINDER_ENERGY_COST_MULTIPLIER = -0.1

        VOID_SEEKER_TASK = True
        VOID_SEEKER_INIT_SCORE = 1200
        VOID_SEEKER_PATH_LENGTH_MULTIPLIER = -5
        VOID_SEEKER_ENERGY_COST_MULTIPLIER = -0.2

        CONTROL_TASK = True
        CONTROL_REWARD_SCORE = 100
        CONTROL_INIT_SCORE = 800
        CONTROL_PATH_LENGTH_MULTIPLIER = -5
        CONTROL_ENERGY_COST_MULTIPLIER = -0.2
        CONTROL_NODE_ENERGY_MULTIPLIER = 5
        CONTROL_MIDDLE_LANE_DISTANCE_MULTIPLIER = 0

        HEAL_TASK = True
        HEAL_NEAR_REWARDS = True
        HEAL_INIT_SCORE = 600
        HEAL_OPP_SPAWN_DISTANCE_MULTIPLIER = -1
        HEAL_SHIP_ENERGY_MULTIPLIER = -1

        MSG_TASK = False
        MSG_TASK_STARTED = False
        MSG_TASK_FINISHED = False

    class MatchOverChill(DefaultParams):
        IL = False
        CONTROL_TASK = False
        VOID_SINGER_INIT_SCORE = 1000
        VOID_SINGER_MIDDLE_LANE_DISTANCE_MULTIPLIER = 100

    class GameOverChill(DefaultParams):
        IL = False
        RELIC_FINDER_TASK = False
        VOID_SEEKER_TASK = False
        CONTROL_TASK = False
        HEAL_TASK = False
        HEAL_NEAR_REWARDS = False
        if IS_KAGGLE:
            HEAL_TASK = True
            MSG_TASK = True

    Params = DefaultParams
    HIDDEN_NODE_ENERGY = Params.HIDDEN_NODE_ENERGY

    @classmethod
    def clear(cls):

        cls.UNIT_MOVE_COST = 1
        cls.UNIT_SAP_COST = 30
        cls.UNIT_SAP_RANGE = 3
        cls.UNIT_SENSOR_RANGE = 2
        cls.NEBULA_ENERGY_REDUCTION = 10
        cls.OBSTACLE_MOVEMENT_PERIOD = 20
        cls.OBSTACLE_MOVEMENT_DIRECTION = (0, 0)
        cls.UNIT_SAP_DROPOFF_FACTOR = 0.5
        cls.UNIT_ENERGY_VOID_FACTOR = 0.125

        cls.ALL_RELICS_FOUND = False
        cls.ALL_REWARDS_FOUND = False
        cls.NEBULA_ENERGY_REDUCTION_FOUND = False
        cls.OBSTACLE_MOVEMENT_PERIOD_FOUND = False
        cls.OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
        cls.UNIT_SAP_DROPOFF_FACTOR_FOUND = False
        cls.UNIT_ENERGY_VOID_FACTOR_FOUND = False

        cls.NUM_COMPLETED_MATCHES = 0
        cls.NUM_WINS = 0
        cls.POINTS = []
        cls.OPP_POINTS = []

        cls.REWARD_RESULTS = []
        cls.OBSTACLES_MOVEMENT_STATUS = []

        cls.Params = cls.DefaultParams


class Colors:
    red = "\033[91m"
    blue = "\033[94m"
    yellow = "\033[93m"
    green = "\033[92m"
    endc = "\033[0m"


SPACE_SIZE = Global.SPACE_SIZE
SECTOR_SIZE = int((SPACE_SIZE**2 + SPACE_SIZE) / 2)  # 300


def log(*args, level=3):
    # 1 - Error
    # 2 - Info
    # 3 - Debug
    if level <= Global.VERBOSITY:
        file = sys.stderr
        if level == 1:
            print(f"{Colors.red}Error{Colors.endc}:", *args, file=file)
        else:
            print(*args, file=file)


def get_match_step(step: int) -> int:
    return step % (Global.MAX_STEPS_IN_MATCH + 1)


def get_match_number(step: int) -> int:
    return step // (Global.MAX_STEPS_IN_MATCH + 1)


def is_upper_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 <= y


def is_middle_line(x, y) -> bool:
    return SPACE_SIZE - x - 1 == y


def is_team_sector(team_id, x, y) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)


def get_opposite(x, y) -> tuple:
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_inside(x, y) -> bool:
    return 0 <= x < SPACE_SIZE and 0 <= y < SPACE_SIZE


def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev_distance(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def nearby_positions(x, y, distance):
    for _x in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for _y in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield _x, _y


def cardinal_positions(x, y):
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        _x = x + dx
        _y = y + dy
        if is_inside(_x, _y):
            yield _x, _y


def get_spawn_location(team_id):
    return (0, 0) if team_id == 0 else (SPACE_SIZE - 1, SPACE_SIZE - 1)


def warp_int(x):
    if x >= SPACE_SIZE:
        x -= SPACE_SIZE
    elif x < 0:
        x += SPACE_SIZE
    return x


def warp_point(x, y):
    return warp_int(x), warp_int(y)


def set_game_prams(new_params):
    if Global.Params == new_params:
        return

    log(f"Update Params {Global.Params.__name__}->{new_params.__name__}", level=2)
    Global.Params = new_params


def transpose(arr, reflective=False):
    if len(arr.shape) == 2:
        if not reflective:
            return arr.T
        else:
            return arr[::-1, ::-1].T

    elif len(arr.shape) == 3:
        if not reflective:
            return np.transpose(arr, axes=[0, 2, 1])
        else:
            return np.transpose(arr[:, ::-1, ::-1], axes=[0, 2, 1])

    else:
        raise ValueError(f"Can't transpose array with the shape {arr.shape}")


def get_nebula_tile_drift_speed():
    speed = 1 / Global.OBSTACLE_MOVEMENT_PERIOD
    if Global.OBSTACLE_MOVEMENT_DIRECTION[0] < 0:
        speed *= -1
    return speed


def elements_moving(step, movement_period=None):
    speed = 1 / movement_period
    return (step - 2) * speed % 1 > (step - 1) * speed % 1


def obstacles_moving(step):
    if not Global.OBSTACLE_MOVEMENT_PERIOD_FOUND:
        return
    return elements_moving(step, movement_period=Global.OBSTACLE_MOVEMENT_PERIOD)


class Task:
    def __init__(self, target):
        self.target = target
        self.ship = None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target})"

    def completed(self, state, ship):
        return False

    def evaluate(self, state, ship):
        return 0

    def apply(self, state, ship):
        pass
