from dataclasses import dataclass

import numpy as np


@dataclass
class GOBnechCfg:
    hmm_length: int = 20
    grid_w: int = 20
    grid_h: int = 1
    M: int = 200
    seed: int = 1

    def pkl_path(self):
        return f"go_exp/length-{self.hmm_length}_gridw-{self.grid_w}_gridh-{self.grid_h}_M-{self.M}_seed-{self.seed}"

    def pkl_name(self):
        return "builder"

    def construct_init(self):
        init = np.array([(self.grid_w - i) for i in range(self.grid_w)])
        return init / np.sum(init)

    def construct_shift_trans(self):
        ones = np.eye(self.grid_w)
        shift = np.hstack([ones[:, 1:], ones[:, 0:1]])
        return shift

    def construct_gaussian_emission(self):
        loc = 5 * np.arange(self.grid_w)
        scale = [1] * self.grid_w
        return {"loc": np.array(loc), "scale": np.array(scale)}
