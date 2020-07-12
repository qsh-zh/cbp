from collections import namedtuple
import numpy as np
from numba import jitclass, types, typed

spec = [
    ('sender', types.unicode_type),
    ('val', types.float64[:, :])
]


# @jitclass(spec)
# class Message(object):
#     def __init__(self, sender, val):
#         self.sender = sender
#         self.val = val / np.sum(val)


# class MailBox(object):
#     def __init__(self):
#         self.record = types(*kv_record)

#     def prod_message(self):
#         box = typed.List.empty_list(types.float64[:, :])
#         for _, val in self.record.item():
#             box.append(val)
#         return np.prod(box)


class Message(namedtuple('Message', ['sender', 'val'])):
    def __new__(cls, sender, val):
        return super(Message, cls).__new__(cls, sender, val / np.sum(val))
