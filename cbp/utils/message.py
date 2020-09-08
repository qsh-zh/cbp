from collections import namedtuple
import numpy as np


class Message(namedtuple('Message', ['sender', 'val'])):
    def __new__(cls, sender, val):
        val = np.clip(val / np.sum(val), 1e-12, None)
        return super(Message, cls).__new__(cls, sender, val)
