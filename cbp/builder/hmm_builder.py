from .base_builder import BaseBuilder


class HMMBuilder(BaseBuilder):
    def __init__(self, length, d, policy, rand_seed=1):
        self.hmm_length = length
        super().__init__(d, policy, rand_seed)

    def init_graph(self):
        self.add_trivial_node()
        self.add_branch(is_constrained=True, is_conv=True)

        for i in range(self.hmm_length - 1):
            self.add_branch(head_node=f"VarNode_{2*i:03d}")
            self.add_branch(is_constrained=True, is_conv=True)
