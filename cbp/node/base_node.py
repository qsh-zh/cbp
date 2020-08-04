import uuid
from abc import ABC
import numpy as np


class BaseNode(ABC):
    """All kinds node must inherit :class:`~cbp.node.BaseNode`. It
    provide the basic register connection functions. Important attr.
        * ``name`` str: id for the node
        * ``connections`` list of str: a list of connected node names
        * ``connected_nodes`` map of nodes: name -> node
        * ``parent`` traverse purpose
    """

    def __init__(self) -> None:
        """Initialize default attr
        """
        self.name = str(uuid.uuid4())
        self.node_degree = 0
        self.parent = None
        self.connections = []
        self.connected_nodes = {}

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

    def format_name(self, name):
        self.name = name

    def check_before_run(self, node_map):
        for item in self.connections:
            assert item in node_map, f"{self.name} has a connection {item}, \
                                                    which is not in node_map"

    def register_connection(self, node_name):
        assert isinstance(node_name, str)
        self.node_degree += 1
        self.connections.append(node_name)

    def register_nodes(self, node_map):
        """initial `connected_nodes`. After build a graph, then do this work

        :param node_map: all nodes in a graph. name -> node
        :type node_map: dict
        :raises IOError: the input graph structure is wrong
        """
        for item in self.connections:
            if item in node_map:
                self.connected_nodes[item] = node_map[item]
            else:
                raise IOError(f"connection of {item} of {self.name} \
                                do not appear in the node_map")

    def search_node_index(self, node_name):
        return self.connections.index(node_name)

    def plot(self, graph):
        graph.add_node(self.name, color='blue', style='bold')

    def __eq__(self, value):
        flag = []
        flag.append(set(self.connections) == set(value.connections))
        flag.append(self.node_degree == value.node_degree)
        if np.sum(flag) == len(flag):
            return True

        return False
