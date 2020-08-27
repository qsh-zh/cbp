from abc import abstractmethod
import numpy as np
from .msg_node import MsgNode


class CnpNode(MsgNode):
    """cnp-type node, set and calculate coef
    """

    def __init__(self, node_coef, potential):
        super().__init__(potential)
        self.node_coef = node_coef
        self.coef_ready = False
        self.epsilon = 1

    def auto_coef(self, node_map, assign_policy=None):
        if assign_policy is None:
            self.node_coef = 1.0 / len(node_map)
        else:
            self.node_coef = assign_policy(self, node_map)

        self.register_nodes(node_map)

    def cal_cnp_coef(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} is an abstract class")

    @abstractmethod
    def cal_bethe(self, margin) -> float:
        """calculate the bethe energy

        :return: bethe energy on this node
        :rtype: float
        """

    def make_init_message(self, recipient_node_name):
        if self.coef_ready:
            recipient_node = self.connected_nodes[recipient_node_name]
            message_dim = recipient_node.potential.shape
            return np.ones(message_dim)

        raise RuntimeError(
            f"Need to call cal_cnp_coef first for {self.name}")

    def __eq__(self, value):
        flag = []
        flag.append((np.isclose(self.node_coef, value.node_coef)).all())
        if np.sum(flag) == len(flag):
            return super().__eq__(value)

        return False
