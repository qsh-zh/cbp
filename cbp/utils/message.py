from collections import namedtuple
import numpy as np


class Message(namedtuple('Message', ['sender', 'val'])):
    def __new__(cls, sender, val):
        return super(Message, cls).__new__(cls, sender, val / np.sum(val))
