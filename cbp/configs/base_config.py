from dataclasses import dataclass


@dataclass
class BaseConfig:
    verbose_send_msg_name: bool = False
    verbose_send_msg_data: bool = False
    verbose_engine_cnp: bool = False
    verbose_itsbp_outer: bool = False
    verbose_itsbp_link: bool = False

    cnp_engine_tolerance: float = 1e-5
    cnp_engine_iteration: float = 5000000

    itsbp_outer_tolerance: float = 1e-4
    itsbp_outer_iteration: float = 5000000

    @staticmethod
    def itsbp_schedule(cnt, leaf_nodes):
        cnt += 1
        return cnt % len(leaf_nodes)


baseconfig = BaseConfig()
