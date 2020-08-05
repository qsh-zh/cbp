from .base_builder import BaseBuilder


class HMMBuilder(BaseBuilder):
    def __init__(self, length, node_dim, policy, rand_seed=1):
        self.hmm_length = length
        self.__step = 0
        super().__init__(node_dim, policy, rand_seed)

    def step(self, time_stamp=None):
        """hmm go forward a step, time + 1

        :param time_stamp: timer for the current add node
        :type time_stamp: int
        """
        if time_stamp is None:
            time_stamp = self.__step
        assert time_stamp > 0
        self.add_branch(head_node=f"VarNode_{2*time_stamp-2:03d}")
        self.add_branch(is_constrained=True, is_obser=True)
        self.__step += 1

    def init_graph(self):
        self.add_trivial_node()
        self.add_branch(is_constrained=True, is_obser=True)
        self.__step = 1

        for i in range(1, self.hmm_length):
            self.step(time_stamp=i)
