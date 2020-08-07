import paperkit as ppk
from go_bench_cfg import GOBnechCfg
from go_bench_executor import GOBenchExecutor

cmdhelper = ppk.CmdHelper(GOBnechCfg)

cmds = ppk.param_sweep(
    prefix=None,
    hmm_length=range(2, 100, 2),
    seed=range(10))

data = {"go_gaussian": {}}

for item in cmds:
    parser = cmdhelper.cfg_from_str(item)
    if parser.hmm_length not in data["go_gaussian"]:
        data["go_gaussian"][parser.hmm_length] = []

    executor = GOBenchExecutor.load(
        f"{parser.pkl_path()}/{parser.pkl_name()}.pkl")
    data["go_gaussian"][parser.hmm_length].append(executor.exp_record["step"])

ax = ppk.meanstd_plot(data, {"go_gaussian": ""}, is_label=False)
ax.figure.savefig('test.png')


def draw(cmds, exp_vary, metric, fig_name):
    data = {"go_gaussian": {}}

    for item in cmds:
        parser = cmdhelper.cfg_from_str(item)
        if parser.hmm_length not in data["go_gaussian"]:
            data["go_gaussian"][getattr(parser, exp_vary)] = []

        executor = GOBenchExecutor.load(
            f"{parser.pkl_path()}/{parser.pkl_name()}.pkl")
        data["go_gaussian"][parser.hmm_length].append(
            executor.exp_record[metric])

    ax = ppk.meanstd_plot(data, {"go_gaussian": ""}, is_label=False)
    ax.figure.savefig(fig_name)


draw(cmds, 'hmm_length', 'step', 'length_step.png')
