import uuid
from abc import ABC, abstractmethod
import numpy as np
from cbp.utils.message import Message


class BaseNode(ABC):
    """All kinds node must inherit :class `~cbp.node.BaseNode`
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, node_coef, potential) -> None:
        """Initialize default attr

        Every node need to have the following attr:

          * ``name`` str. id for the node

        :param node_coef: works for the norm-product algorithm
        :type node_coef: float
        :param potential: potential
        :type potential: ndarray or list
        """
        self.name = str(uuid.uuid4())
        self.node_coef = node_coef
        self._potential = None
        self.potential = potential
        self.epsilon = 1
        self.coef_ready = False
        self.is_traversed = False
        self.parent = None
        self.node_degree = 0
        self.connections = []
        self.message_inbox = {}
        self.latest_message = []
        self.connected_nodes = {}

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()

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

    def format_name(self, name):
        self.name = name

    def reset_node_coef(self, coef):
        self.node_coef = coef

    def auto_coef(self, node_map, assign_policy=None):
        if assign_policy is None:
            self.node_coef = 1.0 / len(node_map)
        else:
            self.node_coef = assign_policy(self, node_map)

        self.register_nodes(node_map)

    # TODO:, should remove node_map parameter
    def cal_cnp_coef(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} is an abstract class")

    def check_before_run(self, node_map):
        for item in self.connections:
            assert item in node_map, f"{self.name} has a connection {item}, \
                                                    which is not in node_map"

    def make_init_message(self, recipient_node_name):
        if self.coef_ready:
            recipient_node = self.connected_nodes[recipient_node_name]
            message_dim = recipient_node.potential.shape
            return np.ones(message_dim)

        raise RuntimeError(
            f"Need to call cal_cnp_coef first for {self.name}")

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

    @abstractmethod
    def cal_bethe(self, margin) -> float:
        """calculate the bethe energy

        :return: bethe energy on this node
        :rtype: float
        """

    def send_message(self, recipient_node, is_silent=True):
        val = self.make_message(recipient_node)
        message = Message(self, val)
        recipient_node.store_message(message)
        if not is_silent:
            print(self.name + '->' + recipient_node.name)
            print(message.val)

    def sendin_message(self, is_silent=True):
        for connected_node in self.connected_nodes.values():
            connected_node.send_message(self, is_silent)

    def sendout_message(self, is_silent=True):
        for connected_node in self.connected_nodes.values():
            self.send_message(connected_node, is_silent)

    def register_connection(self, node_name):
        self.node_degree += 1
        self.connections.append(node_name)

    def register_nodes(self, node_map):
        for item in self.connections:
            if item in node_map:
                self.connected_nodes[item] = node_map[item]
            else:
                raise IOError(f"connection of {item} of {self.name} \
                                do not appear in the node_map")

    def get_connections(self):
        return self.connections

    def search_node_index(self, node_name):
        return self.connections.index(node_name)

    def search_msg_index(self, message_list, node_name):
        which_index = [i for i, message in enumerate(message_list)
                       if message.sender.name == node_name]
        if which_index:
            return which_index[0]

        raise RuntimeError(
            f"{node_name} do not appear in {self.name} message")
