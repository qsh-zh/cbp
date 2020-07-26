import warnings

try:
    import pygraphviz  # noqa
except BaseException:
    pygraphviz = None


class BaseGraph():
    def __init__(self):
        self.varnode_recorder = {}
        self.factornode_recorder = {}
        self.leaf_nodes = []
        self.nodes = []
        self.node_recorder = {}
        self.cnt_varnode = 0
        self.cnt_factornode = 0

    def add_varnode(self, node):
        """add one `~cbp.node.VarNode` to this graph, idx follow the increasing
        order

        :param node: one var node
        :type node: var node
        :return: name of var node
        :rtype: str
        """
        varnode_name = f"VarNode_{self.cnt_varnode:03d}"
        node.format_name(varnode_name)
        self.varnode_recorder[varnode_name] = node
        self.node_recorder[varnode_name] = node

        self.cnt_varnode += 1
        return varnode_name

    def add_factornode(self, factornode):
        """add one factor node to the graph
        Do the following tasks

        *set factornode name attr
        * add node to the recorders
        * set connections
        * set parent relation

        :param factornode: one factor node
        :return: name of factor node
        :rtype: str
        """
        factornode_name = f"FactorNode_{self.cnt_factornode:03d}"
        factornode.format_name(factornode_name)
        self.factornode_recorder[factornode_name] = factornode
        self.node_recorder[factornode_name] = factornode

        self.__register_connection(factornode)
        self.__set_parent(factornode)

        self.cnt_factornode += 1
        return factornode_name

    def __register_connection(self, factornode):
        for varnode_name in factornode.connections:
            varnode = self.varnode_recorder[varnode_name]
            varnode.register_connection(factornode.name)

    def __set_parent(self, factornode):
        connections = factornode.connections
        factornode.parent = self.varnode_recorder[connections[0]]
        for varnode_name in connections[1:]:
            varnode = self.varnode_recorder[varnode_name]
            varnode.parent = factornode

    def bake(self):
        self.init_node_list()

    def init_node_list(self):
        factors = list(self.factornode_recorder.values())
        variables = list(self.varnode_recorder.values())
        # in Norm-Product, run factor message first
        self.nodes = factors + variables
        self.leaf_nodes = [
            node for node in self.nodes if len(node.connections) == 1]

    def get_root(self):
        if len(self.leaf_nodes) == 0:
            raise RuntimeError("graph contains circle")
        return self.leaf_nodes[0]

    def get_node(self, name_str):
        if name_str not in self.node_recorder:
            raise RuntimeError(f"{name_str} is illegal, not in this graph")
        return self.node_recorder[name_str]

    def delete_node(self, name_str):
        """delete node from graph, needs to check following

        * clear connections of connected nodes
        * delete from the various recorders
        * call init_node_list

        :param name_str: [description]
        :type name_str: [type]
        :raises RuntimeError: [description]
        """
        if name_str not in self.node_recorder:
            raise RuntimeError(f"{name_str} is illegal, not in this graph")
        target_node = self.node_recorder[name_str]
        if 'var' in name_str.lower():
            warnings.warn(f"Delete {name_str}, may have a suspend factor node")
        for connected_name in target_node.connections:
            connected_node = self.node_recorder[connected_name]
            if connected_node.parent is target_node:
                connected_node.parent = None
            connected_node.connections.remove(name_str)

            # clear map
            if len(connected_node.connections) == 0:
                self._delete_node_recorder(connected_node)

        self._delete_node_recorder(target_node)

    def _delete_node_recorder(self, node):
        target_map = self.varnode_recorder if 'var' in node.name.lower(
        ) else self.factornode_recorder
        del target_map[node.name]
        del self.node_recorder[node.name]

    def plot(self, png_name='file.png'):
        """plot the graph through graphviz

            * red: constrained variable node
            * blue: free variable node
            * green: factor

        :param png_name: name of figure, defaults to 'file.png'
        :type png_name: str, optional
        :raises ValueError: [description]
        """
        if pygraphviz is not None:
            graph = pygraphviz.AGraph(directed=False)
            for node in self.nodes:
                node.plot(graph)

            graph.layout(prog='neato')
            graph.draw(png_name)
        else:
            raise ValueError("must have pygraphviz installed for visualization")
