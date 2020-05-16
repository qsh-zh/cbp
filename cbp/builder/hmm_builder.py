from .base_builder import BaseBuilder


class HMMBuilder(BaseBuilder):
    def __init__(self, length, d, policy, rand_seed=1):
        self.hmm_length = length
        super().__init__(d, policy, rand_seed)

    def step(self, time_stamp):
        """hmm go forward a step, time + 1

        :param time_stamp: timer for the current add node
        :type time_stamp: int
        """
        assert time_stamp > 0
        self.add_branch(head_node=f"VarNode_{2*time_stamp-2:03d}")
        self.add_branch(is_constrained=True, is_conv=True)

    def init_graph(self):
        self.add_trivial_node()
        self.add_branch(is_constrained=True, is_conv=True)

        for i in range(1, self.hmm_length):
            self.step(time_stamp=i)
