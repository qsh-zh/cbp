from dataclasses import dataclass


@dataclass
class BaseConfig:

    @staticmethod
    def itsbp_schedule(cnt, leaf_nodes):
        cnt += 1
        return cnt % len(leaf_nodes)


baseconfig = BaseConfig()
