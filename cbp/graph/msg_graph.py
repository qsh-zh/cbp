from .constrain_graph import ConstrainGraph
from cbp.utils import (compare_marginals, diff_max_marginals,
                       engine_loop, Message)

from .graph_utils import itsbp_inner_loop, find_link
from cbp.configs.base_config import baseconfig


class MsgGraph(ConstrainGraph):
    """implement the basic msg passing schedule and itsbp methods
    """

    def __init__(self, config=baseconfig):
        super().__init__()
        self.cfg = config
        self.itsbp_outer_cnt = 0

    def engine_loop(  # pylint: disable= too-many-arguments
            self,
            engine_fun,
            max_iter=5000000,
            tolerance=1e-2,
            error_fun=None,
            isoutput=False):
        if error_fun is None:
            error_fun = compare_marginals

        epsilons, step, timer = engine_loop(
            engine_fun=engine_fun,
            max_iter=max_iter,
            tolerance=tolerance,
            error_fun=error_fun,
            meassure_fun=self.export_convergence_marginals,
            isoutput=isoutput)

        return epsilons, step, timer

    def itsbp(self):
        """run sinkhorn or iterative scaling inference

        :return: [description]
        :rtype: [type]
        """
        self.first_belief_propagation()
        return self.engine_loop(self.itsbp_outer_loop,
                                tolerance=self.cfg.itsbp_outer_tolerance,
                                error_fun=diff_max_marginals,
                                isoutput=self.cfg.verbose_itsbp_outer)

    def its_next_looplink(self):
        target_node = self.leaf_nodes[self.itsbp_outer_cnt]

        next_node = self.leaf_nodes[(
            self.itsbp_outer_cnt + 1) % len(self.leaf_nodes)]

        self.itsbp_outer_cnt = self.cfg.itsbp_schedule(
            self.itsbp_outer_cnt, self.leaf_nodes)
        return target_node, find_link(target_node, next_node)

    def itsbp_outer_loop(self):
        for _ in range(len(self.leaf_nodes)):
            _, loop_link = self.its_next_looplink()
            itsbp_inner_loop(loop_link, self.cfg.verbose_node_send_msg)

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

    def tree_bp(self):
        """run classical belief propagation on a tree graph, only need forward
        and backward

            * add attr: is_send_forward: begin send forward false, after forward
             before backward true, after backward false
        :raises RuntimeError: Only works for the tree graph, loopy graph does
        not work, root node not decided
        """
        self.bake()
        self.first_belief_propagation()
        for node in self.nodes:
            setattr(node, 'is_send_forward', False)

        tree_root = self.get_root()

        self._send_forward(tree_root)
        self._send_backward(tree_root)

    def _send_forward(self, node):
        node.is_send_forward = True
        for cur_node in node.connected_nodes.values():
            if not cur_node.is_send_forward:
                self._send_forward(cur_node)
                cur_node.send_message(node)

    def _send_backward(self, node):
        node.is_send_forward = False
        for cur_node in node.connected_nodes.values():
            if cur_node.is_send_forward:
                node.send_message(cur_node)
                self._send_backward(cur_node)

    def first_belief_propagation(self):
        for node in self.nodes:
            for recipient_name in node.connections:
                recipient = self.node_recorder[recipient_name]
                if node.name not in recipient.message_inbox:
                    val = node.make_init_message(recipient_name)
                    message = Message(node, val)
                    self.node_recorder[recipient_name].store_message(message)

    def parallel_message(self, run_constrained=True):
        for target_var in self.varnode_recorder.values():
            # sendind in messages from factors
            target_var.sendin_message(self.cfg.verbose_node_send_msg)

            if run_constrained or (not target_var.isconstrained):
                target_var.sendout_message(self.cfg.verbose_node_send_msg)

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
