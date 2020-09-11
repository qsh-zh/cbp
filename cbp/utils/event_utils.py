import time

import numpy as np


def engine_loop(  # pylint: disable=too-many-arguments
        engine_fun,
        max_iter=5000000,
        tolerance=1e-2,
        error_fun=None,
        meassure_fun=None,
        isoutput=False):
    check_step = 1
    epsilons = [np.inf] * check_step
    start = time.time()
    timer_record = []
    step = 0
    cur_marginals = meassure_fun()

    while (step < max_iter) and any(
            tolerance < np.array(epsilons[-check_step:])):
        last_marginals = cur_marginals
        step += 1

        engine_fun()
        cur_marginals = meassure_fun()
        epsilons.append(error_fun(cur_marginals, last_marginals))

        timer_record.append(time.time() - start)
        if isoutput:
            print(f'step: {step:5d} | verobose output {epsilons[-1]}')

    return epsilons[check_step:], step, timer_record


def compare_marginals(mar_1, mar_2):
    assert np.setdiff1d(mar_1.keys(), mar_2.keys()).size == 0
    return sum([np.sum(np.absolute(mar_1[k] - mar_2[k])) for k in mar_1.keys()])


def diff_max_marginals(mar_1, mar_2):
    assert np.setdiff1d(mar_1.keys(), mar_2.keys()).size == 0
    return np.max([np.sum(np.absolute(mar_1[k] - mar_2[k]))
                   for k in mar_1.keys()])
