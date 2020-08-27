from abc import abstractmethod

import numpy as np
from cbp.utils.message import Message

from .base_node import BaseNode


class MsgNode(BaseNode):
    """message passing interfaces
    """

    def __init__(self, potential):
        """[summary]

        :param node_coef: works for the norm-product algorithm
        :type node_coef: float
        :param potential: potential
        :type potential: ndarray or list
        """
        super().__init__()
        self._potential = None
        self.potential = potential
        self.message_inbox = {}
        self.latest_message = []

    @property
    def potential(self):
        return self._potential

    @potential.setter
    def potential(self, potential):
        self._potential = self._check_potential(potential)

    @abstractmethod
    def _check_potential(self, potential) -> np.ndarray:
        """check potential before set node potential

        :param potential: input potential
        :type potential: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """

    @abstractmethod
    def make_init_message(self, recipient_node_name):
        """make init messsage to neighbor

        :param recipient_node_name: neighbor node
        :type recipient_node_name: str
        """

    # keep all message looks urgly. convenient for debug and resource occupied
    # is not so huge
    def store_message(self, message):
        sender_name = message.sender.name
        self.message_inbox[sender_name] = message

        self.latest_message = list(self.message_inbox.values())

    def reset(self):
        self.message_inbox.clear()

    # TODO: FIXAPI NAME
    @abstractmethod
    def make_message(self, recipient_node) -> np.ndarray:
        """produce the val of message from current node to the recipient_node

        :param recipient_node: target node
        :type recipient_node: [type]
        :return: content of the message
        :rtype: np.ndarray
        """

    def send_message(self, recipient_node, verbose=False):
        """send message from this node to target node

        :param recipient_node: target node
        :param verbose: debug, defaults to False
        :type verbose: bool, optional
        """
        val = self.make_message(recipient_node)
        message = Message(self, val)
        recipient_node.store_message(message)
        if verbose:
            print(self.name + '->' + recipient_node.name)
            print(message.val)

    def sendin_message(self, verbose=False):
        for connected_node in self.connected_nodes.values():
            connected_node.send_message(self, verbose)

    def sendout_message(self, verbose=False):
        for connected_node in self.connected_nodes.values():
            self.send_message(connected_node, verbose)

    def search_msg_index(self, message_list, node_name):
        which_index = [i for i, message in enumerate(message_list)
                       if message.sender.name == node_name]
        if which_index:
            return which_index[0]

        raise RuntimeError(
            f"{node_name} do not appear in {self.name} message")

    def __eq__(self, value):
        flag = []
        flag.append((np.isclose(self.node_coef, value.node_coef)).all())
        flag.append(np.isclose(self.potential, value.potential).all())
        if np.sum(flag) == len(flag):
            return super().__eq__(value)

        return False
