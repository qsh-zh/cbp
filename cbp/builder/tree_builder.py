from .base_builder import BaseBuilder


class TreeBuilder(BaseBuilder):
    def __init__(self, tree_depth, d, policy, rand_seed=1):
        self.tree_depth = tree_depth
        super().__init__(d, policy, rand_seed)

    def init_graph(self):
        self.add_trivial_node()
        for depth in range(self.tree_depth - 2):
            for cur_node in range(2**depth - 1, 2**(depth + 1) - 1):
                for _ in range(2):
                    self.add_branch(head_node=f"VarNode_{cur_node:03d}")

        depth = self.tree_depth - 2
        for cur_node in range(2**depth - 1, 2**(depth + 1) - 1):
            for _ in range(2):
                self.add_branch(head_node=f"VarNode_{cur_node:03d}",
                                is_constrained=True)
