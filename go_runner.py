from go_bench_cfg import GOBnechCfg
from go_bench_executor import GOBenchExecutor

import paperkit as ppk

cmds = ppk.param_sweep(
    prefix="python go_bench_executor.py",
    seed=range(7))

ppk.bash_execute(cmds, num_thread=10, msg="go KL")

