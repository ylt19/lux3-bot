from .space import Node


class Task:
    def __repr__(self):
        return self.__class__.__name__


class FindRelicNodes(Task):
    pass


class FindRewardNodes(Task):
    def __init__(self, node: Node):
        assert node.relic
        self.coordinates = node.x, node.y

    def __repr__(self):
        return f"FindRewardNodes{self.coordinates}"


class HarvestTask(Task):
    def __init__(self, node: Node):
        assert node.reward
        self.coordinates = node.x, node.y

    def __repr__(self):
        return f"HarvestTask{self.coordinates}"
