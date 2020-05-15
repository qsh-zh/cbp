from .base_builder import BaseBuilder


class LineBuilder(BaseBuilder):
    def __init__(self, num_node, d, policy, rand_seed=1):
        self.num_node = num_node
        super().__init__(d, policy, rand_seed)

    def init_graph(self):
        self.add_constrained_node()
        for _ in range(self.num_node - 2):
            self.add_branch(is_constrained=False)
        self.add_branch(is_constrained=True)
