import paperkit as ppk
from go_bench_cfg import GOBnechCfg
from go_bench_executor import GOBenchExecutor

import matplotlib.pyplot as plt
plt.style.use('paper')

cmdhelper = ppk.CmdHelper(GOBnechCfg)

cmds = ppk.param_sweep(
    prefix=None,
    hmm_length=range(2, 90, 2),
    seed=range(10))


def draw(cmds, exp_vary, metric, fig_name, **kwargs):
    data = {"go_gaussian": {}}

    for item in cmds:
        parser = cmdhelper.cfg_from_str(item)
        if getattr(parser, exp_vary) not in data["go_gaussian"]:
            data["go_gaussian"][getattr(parser, exp_vary)] = []

        executor = GOBenchExecutor.load(
            f"{parser.pkl_path()}/{parser.pkl_name()}.pkl")
        data["go_gaussian"][getattr(parser, exp_vary)].append(
            executor.exp_record[metric])

    ax = ppk.meanstd_plot(data,
                          {"go_gaussian": ""},
                          is_label=False,
                          **kwargs)
    ax.figure.savefig(fig_name)


# draw(
#     cmds,
#     'hmm_length',
#     'step',
#     'length_step.png',
#     xlabel="T",
#     ylabel="Iteration")
# draw(
#     cmds,
#     'hmm_length',
#     'time',
#     'length_time.png',
#     xlabel="T",
#     ylabel="Time(second)")

cmdhelper = ppk.CmdHelper(GOBnechCfg)
cmds = ppk.param_sweep(
    prefix=None,
    grid_w=range(20, 200, 5),
    seed=range(10))
draw(
    cmds,
    'grid_w',
    'step',
    'state_step.png',
    xlabel="d",
    ylabel="Iteration")
draw(
    cmds,
    'grid_w',
    'time',
    'state_time.png',
    xlabel="d",
    ylabel="Time(second)")
