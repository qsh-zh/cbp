import paperkit as ppk
from go_bench_cfg import GOBnechCfg
from go_bench_executor import GOBenchExecutor

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('paper')

cmdhelper = ppk.CmdHelper(GOBnechCfg)
cmds = ppk.param_sweep(seed=range(1, 7))


def KL_draw(cmds):
    data = []
    for item in cmds:
        parser = cmdhelper.cfg_from_str(item)
        executor = GOBenchExecutor.load(
            f"{parser.pkl_path()}/{parser.pkl_name()}.pkl")
        data.append(
            (executor.exp_record["step"],
             executor.exp_record["record_KL"]))

    fig, ax = plt.subplots(figsize=(7, 7))
    for step, kl in data:
        kl = kl[:100]
        x = np.arange(1, len(kl) + 1)
        ax.plot(x, kl)

    ax.set_xlabel("Iteration Step")
    ax.set_ylabel("Sum of L1 Error")
    ax.set_yscale("log")
    fig.savefig("error.png")


if __name__ == "__main__":
    KL_draw(cmds)
