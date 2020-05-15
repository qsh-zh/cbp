from .base_builder import BaseBuilder


class StarBuilder(BaseBuilder):
    def __init__(self, num_node, d, policy, rand_seed):
        self.num_node = num_node
        super().__init__(d, policy, rand_seed)

    def init_graph(self):
        center_node = "VarNode_000"
        self.add_trivial_node()

        for _ in range(self.num_node - 1):
            self.add_branch(head_node=center_node, is_constrained=True)
