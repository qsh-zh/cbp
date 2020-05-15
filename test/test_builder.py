import unittest
from pathlib import Path
from cbp import builder
from cbp.graph.coef_policy import bp_policy


class TestPlotGraph(unittest.TestCase):
    def test_plot_graph(self):
        Path('data').mkdir(exist_ok=True)
        builder_func = {
            "line": builder.LineBuilder,
            "star": builder.StarBuilder,
            "hmm": builder.HMMBuilder,
            "tree": builder.TreeBuilder
        }
        for name, worker in builder_func.items():
            graph = worker(4, 4, bp_policy, rand_seed=1)()
            graph.plot(f"data/{name}.png")
