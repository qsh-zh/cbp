import numpy as np


class Message():  # pylint: disable=too-few-public-methods
    def __init__(self, sender, val):
        self.sender = sender
        # FIXME divide by zero or nan
        val = np.nan_to_num(val)
        partion = np.sum(val.flatten())
        self.val = val / partion
