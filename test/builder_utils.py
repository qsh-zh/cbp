from pathlib import Path

import numpy as np
from numpy.random import RandomState
from cbp.graph import GraphModel
from cbp.node import FactorNode, VarNode


def two_node_tree():
    graph = GraphModel()

    # init varnode
    for _ in range(2):
        varnode = VarNode(2)
        graph.add_varnode(varnode)
    connect_var = ['VarNode_000', 'VarNode_001']
    factor_potential = np.array([
        [3, 2],
        [3, 17]
    ])
    factornode = FactorNode(connect_var, factor_potential)
    graph.add_factornode(factornode)

    return graph


def three_node_tree():
    graph = two_node_tree()
    varnode = VarNode(2)
    graph.add_varnode(varnode)
    connect_var = ['VarNode_001', 'VarNode_002']
    factor_potential = np.array([
        [6, 14],
        [1, 4]
    ])
    factornode = FactorNode(connect_var, factor_potential)
    graph.add_factornode(factornode)

    return graph


def six_node_graph():
    Path('data').mkdir(exist_ok=True)
    rng = RandomState(1)
    graph = GraphModel()
    rv_dim = 2

    # init varnode
    for _ in range(6):
        potential = np.ones([rv_dim])
        varnode = VarNode(rv_dim, potential)
        graph.add_varnode(varnode)

    # init factornode
    edges = [
        [0, 1],
        [2, 1],
        [1, 3],
        [3, 4],
        [3, 5]
    ]
    for item in edges:
        potential = rng.normal(size=[rv_dim, rv_dim])
        factorname = [f"VarNode_{data:03d}" for data in item]
        factornode = FactorNode(factorname, np.exp(potential))
        graph.add_factornode(factornode)
    graph.plot(f"data/six_node_graph.png")

    return graph
