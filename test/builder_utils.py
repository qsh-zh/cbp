import numpy as np
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
