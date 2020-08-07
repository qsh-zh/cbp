from go_bench_cfg import GOBnechCfg
from go_bench_executor import GOBenchExecutor

import paperkit as ppk

cmds = ppk.param_sweep(
    prefix="python go_bench_executor.py",
    hmm_length=range(2, 100, 2))

ppk.bash_execute(cmds, num_thread=10, msg="go vary lenght")


cmds = ppk.param_sweep(
    prefix="python go_bench_executor.py",
    grid_w=range(20, 200, 5))

ppk.bash_execute(cmds, num_thread=10, msg="go vary lenght")
