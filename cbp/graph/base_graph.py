import warnings
from functools import partial

import numpy as np
from cbp.node import VarNode
from cbp.utils import (Message, diff_max_marginals,
                       engine_loop)
from cbp.utils.np_utils import (nd_expand, nd_multiexpand,
                                reduction_ndarray)
from cbp.configs.base_config import baseconfig
from .coef_policy import bp_policy
from .graph_utils import cal_marginal_from_tensor
try:
    import pygraphviz  # noqa
except BaseException:
    pygraphviz = None


class BaseGraph():  # pylint: disable=too-many-instance-attributes
    def __init__(self, silent=True, epsilon=1,
                 coef_policy=bp_policy, config=baseconfig):
        self.varnode_recorder = {}
        self.constrained_names = []
        self.leaf_nodes = []
        self.factornode_recorder = {}
        self.node_recorder = {}
        self.epsilon = epsilon
        self.coef_policy = coef_policy
        self.cnt_varnode = 0
        self.cnt_factornode = 0
        self.cfg = config

        # debug utils
        self.silent = silent

    def add_varnode(self, node):
        """add one `~cbp.node.VarNode` to this graph, idx follow the increasing
        order

        :param node: one VarNode
        :type node: VarNode
        :return: name of varnode
        :rtype: str
        """
        assert isinstance(node, VarNode)
        varnode_name = f"VarNode_{self.cnt_varnode:03d}"
        node.format_name(varnode_name)
        self.varnode_recorder[varnode_name] = node
        self.node_recorder[varnode_name] = node
        if node.isconstrained:
            self.constrained_names.append(varnode_name)

        self.cnt_varnode += 1
        return varnode_name

    def add_factornode(self, factornode):
        """add one factor node to the graph
        Do the following tasks

        * add node to the recorders
        * set connections
        * set parent relation

        :param factornode: one factor node
        :type factornode: FactorNode
        :return: name of factor node
        :rtype: str
        """
        factornode.check_potential(self.varnode_recorder)
        factornode_name = f"FactorNode_{self.cnt_factornode:03d}"
        factornode.format_name(factornode_name)
        self.factornode_recorder[factornode_name] = factornode
        self.node_recorder[factornode_name] = factornode

        self.__register_connection(factornode)
        self.__set_parent(factornode)

        self.cnt_factornode += 1
        return factornode_name

    def __register_connection(self, factornode):
        for varnode_name in factornode.get_connections():
            varnode = self.varnode_recorder[varnode_name]
            varnode.register_connection(factornode.name)

    def __set_parent(self, factornode):
        connections = factornode.get_connections()
        factornode.parent = self.varnode_recorder[connections[0]]
        for varnode_name in connections[1:]:
            varnode = self.varnode_recorder[varnode_name]
            varnode.parent = factornode

    def pmf(self):
        """output the probability mass matrix through brutal-force methods

        :return: joint probability mass matrix
        :rtype: ndarray
        """
        varnode_names = list(self.varnode_recorder.keys())
        varnodes = list(self.varnode_recorder.values())
        var_dim = [variable.rv_dim for variable in varnodes]
        assert len(var_dim) < 32, "max number of vars for brute_force is 32 \
            (numpy matrix dim limit)"
        joint_acc = np.ones(var_dim)
        for factor in self.factornode_recorder.values():
            which_dims = [varnode_names.index(v)
                          for v in factor.get_connections()]
            factor_acc = np.ones(var_dim)

            factor_acc = nd_multiexpand(factor.potential, var_dim, which_dims)

            joint_acc *= factor_acc

        joint_prob = joint_acc / np.sum(joint_acc)
        return joint_prob

    def exact_marginal(self):
        varnodes = list(self.varnode_recorder.values())
        prob_tensor = self.pmf()

        marginal_list = cal_marginal_from_tensor(prob_tensor, varnodes)
        for node, marginal in zip(varnodes, marginal_list):
            node.bfmarginal = marginal

    def first_belief_propagation(self):
        for node in self.nodes:
            for recipient_name in node.connections:
                recipient = self.node_recorder[recipient_name]
                if recipient.name not in node.message_inbox:
                    val = node.make_init_message(recipient_name)
                    message = Message(node, val)
                    self.node_recorder[recipient_name].store_message(message)

    def copy_bp_initialization(self, another_graph):
        """copy message setup from the another graph has same structure

        :param another_graph: another graph which close to the optimal point
        :type another_graph: BaseGraph
        """
        # TODO: copy safety!!!
        for node in self.nodes:
            if node.name in another_graph.node_recorder:
                another_node = another_graph.node_recorder[node.name]
                node.message_inbox = another_node.message_inbox
                node.latest_message = another_node.latest_message
            else:
                raise f"{node.name} not in this graph"

    def __init_sinkhorn_node(self):
        varnode_names = list(self.varnode_recorder.keys())
        self.sinkhorn_node_coef = {}  # pylint: disable=attribute-defined-outside-init
        for node_name in self.constrained_names:
            node_instance = self.varnode_recorder[node_name]
            self.sinkhorn_node_coef[node_name] = {
                'index': varnode_names.index(node_name),
                'mu': node_instance.constrained_marginal,
                'u': np.ones(node_instance.rv_dim)
            }
        for node in self.varnode_recorder.values():
            node.sinkhorn = np.ones(node.rv_dim) / node.rv_dim

    def __build_big_u(self):
        varnodes = list(self.varnode_recorder.values())
        var_dim = [variable.rv_dim for variable in varnodes]
        joint_acc = np.ones(var_dim)

        for _, recoder in self.sinkhorn_node_coef.items():
            constrained_acc = nd_expand(
                recoder['u'], tuple(var_dim), recoder['index'])
            joint_acc *= constrained_acc

        # log_joint_acc -= np.max(log_joint_acc)
        return joint_acc / np.sum(joint_acc)

    # TODO this is a bug!!!!
    def sinkhorn_update(self, tilde_c):
        for _, recorder in self.sinkhorn_node_coef.items():
            big_u = self.__build_big_u()
            normalized_denominator = (big_u * tilde_c) / \
                np.sum(big_u * tilde_c)

            copy_denominator = reduction_ndarray(
                normalized_denominator, recorder['index'])
            copy_denominator = np.clip(copy_denominator, 1e-12, None)
            recorder['u'] = recorder['u'] * recorder['mu'] / copy_denominator

        varnodes = list(self.varnode_recorder.values())
        marginal_list = cal_marginal_from_tensor(
            normalized_denominator, varnodes)
        for node, marginal in zip(varnodes, marginal_list):
            node.sinkhorn = marginal

    def check_sinkhorn(self):
        if len(self.constrained_names) == 0:
            raise RuntimeError(
                "There is no constrained nodes, use brutal force")

    def sinkhorn(self, max_iter=5000000, tolerance=1e-5):
        self.check_sinkhorn()
        tilde_c = self.pmf()
        self.__init_sinkhorn_node()

        sinkhorn_func = partial(self.sinkhorn_update, tilde_c)
        return engine_loop(engine_fun=sinkhorn_func,
                           max_iter=max_iter,
                           tolerance=tolerance,
                           error_fun=diff_max_marginals,
                           meassure_fun=self.export_sinkhorn,
                           isoutput=False,
                           silent=self.silent)

    def bake(self):
        self.init_node_recorder()
        for node in self.nodes:
            if len(node.connections) == 1:
                root = node
        self.auto_node_coef(root)

    def auto_node_coef(self, node):
        node.is_traversed = True
        for item in node.connections:
            if not self.node_recorder[item].is_traversed:
                self.auto_node_coef(self.node_recorder[item])

        node.auto_coef(self.node_recorder, self.coef_policy)
        node.is_traversed = False

    def init_node_recorder(self):
        factors = list(self.factornode_recorder.values())
        variables = list(self.varnode_recorder.values())
        # in Norm-Product, run factor message first
        self.nodes = factors + variables  # pylint: disable=attribute-defined-outside-init
        self.leaf_nodes = [
            node for node in self.nodes if len(node.get_connections()) == 1]

    def get_node(self, name_str):
        if name_str not in self.node_recorder:
            raise RuntimeError(f"{name_str} is illegal, not in this graph")
        return self.node_recorder[name_str]

    def cal_bethe(self, margin):
        """calculate bethe energy

        :param margin: node_name : margin
        :type margin: dict
        :return: KL divergence between expoert joint dist and p_graph
        :rtype: float
        """
        sum_item = []
        for node in self.nodes:
            sum_item.append(node.cal_bethe(margin[node.name]))

        return np.sum(sum_item)

    def delete_node(self, name_str):
        """delete node from graph, needs to check following

        * judge type, Factor or Variable
        * delete from the various recorders
        * clear connections

        :param name_str: [description]
        :type name_str: [type]
        :raises RuntimeError: [description]
        """
        if name_str not in self.node_recorder:
            raise RuntimeError(f"{name_str} is illegal, not in this graph")
        target_node = self.node_recorder[name_str]
        if isinstance(target_node, VarNode):
            warnings.warn(f"Delete {name_str}, may have a suspend factor node")
        for connected_name in target_node.get_connections():
            connected_node = self.node_recorder[connected_name]
            if connected_node.parent is target_node:
                connected_node.parent = None
            connected_node.get_connections().remove(name_str)

            # clear map
            if len(connected_node.get_connections()) == 0:
                self.__delete_node_recorder(connected_node)

        self.__delete_node_recorder(target_node)

    def __delete_node_recorder(self, node):
        target_map = self.varnode_recorder if isinstance(
            node, VarNode) else self.factornode_recorder
        del target_map[node.name]
        del self.node_recorder[node.name]
        if node.name in self.constrained_names:
            self.constrained_names.remove(node.name)

    def set_node(self, node_name, potential=None, isconstrained=None):
        """change node property
        1. check whether or not in recorder
        2. change potential easily[this may a duplicate function]
        3. change isconstrained if possible, delete from recorder

        :param node_name: [description]
        :type node_name: [type]
        :param potential: [description], defaults to None
        :type potential: [type], optional
        :param isconstrained: [description], defaults to None
        :type isconstrained: [type], optional
        :raises RuntimeError: [description]
        """
        if node_name not in self.node_recorder:
            raise RuntimeError
        node = self.node_recorder[node_name]
        if potential is not None:
            # TODO: make potential property check, when do set
            node.potential = potential
        if isconstrained is not None:
            if node.isconstrained != isconstrained:
                node.isconstrained = isconstrained
                if isconstrained:
                    self.constrained_names.append(node_name)
                else:
                    self.constrained_names.remove(node_name)

    def export_marginals(self):
        """export the marginal for variable nodes

        :return: {node.key:node.marginal}
        :rtype: dict
        """
        return {
            n.name: n.marginal() for n in self.varnode_recorder.values()
        }

    def export_convergence_marginals(self):
        """export the marginal for variable nodes and factor nodes

        :return: {node.key:node.marginal}
        :rtype: dict
        """
        return {n.name: n.marginal() for n in self.nodes}

    def export_sinkhorn(self):
        return {node_name: node.sinkhorn
                for node_name, node in self.varnode_recorder.items()}

    def plot(self, png_name='file.png'):
        """plot the graph through graphviz

            * red: constrained variable
            * blue: free variable node
            * green: factor

        :param png_name: name of figure, defaults to 'file.png'
        :type png_name: str, optional
        :raises ValueError: [description]
        """
        if pygraphviz is not None:
            graph = pygraphviz.AGraph(directed=False)
            for varnode_name in self.varnode_recorder:
                if varnode_name in self.constrained_names:
                    graph.add_node(varnode_name, color='red', style='filled')
                else:
                    graph.add_node(varnode_name, color='blue', style='bold')

            for name, factornode in self.factornode_recorder.items():
                graph.add_node(name, color='green')
                for varnode_name in factornode.get_connections():
                    graph.add_edge(name, varnode_name)

            graph.layout(prog='neato')
            graph.draw(png_name)
        else:
            raise ValueError("must have pygraphviz installed for visualization")
