from .space import Node


class Task:
    def __repr__(self):
        return self.__class__.__name__


class FindRelicNodes(Task):
    pass


class FindRewardNodes(Task):
    def __init__(self, node: Node):
        assert not node.explored_for_reward
        self.node = node


class HarvestTask(Task):
    def __init__(self, node: Node):
        assert node.reward
        self.node = node
