from dataclasses import dataclass
from .base_config import BaseConfig


@dataclass
class TestConfig(BaseConfig):
    @staticmethod
    def itsbp_schedule(cnt, leaf_nodes):
        if cnt == len(leaf_nodes) - 2:
            cnt = -1
            leaf_nodes.reverse()
        return cnt + 1
