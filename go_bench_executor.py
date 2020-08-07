from cbp.builder import GOHMMSimulator, HMMGOSimBuilder, PotentialType
from cbp.graph import bp_policy
from paperkit import _Drawer


class GOBenchExecutor(_Drawer):
    def __init__(self, cfg):
        self.cfg = cfg
        sim = self.construct_sim()
        self.drawer_path = self.cfg.pkl_path()
        self._draw_name = self.cfg.pkl_name()
        self.exp_record = {}
        self.builder = self.construct_builder(sim)

    def construct_sim(self):
        sim = GOHMMSimulator(
            self.cfg.hmm_length,
            self.cfg.grid_w * self.cfg.grid_h,
            self.cfg.seed)
        sim.register_potential(PotentialType.INIT, self.cfg.construct_init())
        sim.register_potential(PotentialType.TRANSITION,
                               self.cfg.construct_shift_trans())
        sim.register_potential(PotentialType.EMISSION,
                               self.cfg.construct_gaussian_emission())
        sim.sample(self.cfg.M)
        return sim

    def construct_builder(self, sim):
        builder = HMMGOSimBuilder(self.cfg.hmm_length, sim, bp_policy)
        graph = builder()
        builder.fix_initpotential()
        graph.bake()
        return builder

    def run_exp(self):
        disc_graph = self.builder.graph.discrete_graph
        _, step, timer = disc_graph.run_bp()
        self.exp_record["step"] = step
        self.exp_record["time"] = timer[-1]

        self.save()


if __name__ == "__main__":
    from go_bench_cfg import GOBnechCfg
    from paperkit import CmdHelper
    helper = CmdHelper(GOBnechCfg)
    cfg, _ = helper.cfg_from_cmd()
    worker = GOBenchExecutor(cfg)
    worker.run_exp()
