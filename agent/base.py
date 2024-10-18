SPACE_SIZE = 24
RELIC_REWARD_RANGE = 2
MAX_ENERGY_PER_TILE = 20
MIN_ENERGY_PER_TILE = -20
MAX_STEPS_IN_MATCH = 100


class Step:
    def __init__(self):
        self.global_step = 0
        self.match_step = 0
        self.match = 0

    def update(self):
        self.global_step += 1
        self.match_step += 1
        if self.match_step > MAX_STEPS_IN_MATCH:
            self.match_step = 0
            self.match += 1

    def __repr__(self):
        return f"step {self.global_step}: {self.match}.{self.match_step}"


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


def nearby_positions(x, y, distance):
    for _x in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for _y in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield _x, _y
