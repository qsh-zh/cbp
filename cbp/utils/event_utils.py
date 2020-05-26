import time

import numpy as np


def engine_loop(  # pylint: disable=too-many-arguments
        engine_fun,
        max_iter=5000000,
        tolerance=1e-2,
        error_fun=None,
        meassure_fun=None,
        isoutput=False,
        silent=False):
    """runtine worker, a general interation algorithms

    * Compare relative changed, smaller than the tolerance, stop

    :param max_iter: iteration upper count, defaults to 5000000
    :type max_iter: int, optional
    :param tolerance: to which stop iteration, defaults to 1e-2
    :type tolerance: float32, optional
    :param error_fun: calculate the distance from last measurement and this
    measurement, defaults to None
    :type error_fun: function, optional
    :param meassure_fun: return the marginal of the graph, defaults to None
    :type meassure_fun: function, optional
    :param isoutput: print each step error, defaults to False
    :type isoutput: bool, optional
    :param silent: more detail debug info, defaults to False
    :type silent: bool, optional
    :return: error_list, step_num, elapsed time for each iteration
    :rtype: tuple
    """
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
        if not silent:
            print(
                f"epsilon: {epsilons[-1]:5.4f} | step: {step:5d} {'-'*10}")
            print(cur_marginals)
            print(last_marginals)
            print(epsilons)
        if isoutput:
            print(f'step: {step:5d} | verobose output {epsilons[-1]}')

    return epsilons[check_step:], step, timer_record


def diff_1d_marginal(mar_1, mar_2):
    """For each marginal, the diff is calculated according to 1d-norm,
    return summation of all diff

    :param mar_1: dict of first marginal
    :type mar_1: dict
    :param mar_2: dict of the second marginal
    :type mar_2: dict
    :return: distance
    :rtype: float32
    """
    assert np.setdiff1d(mar_1.keys(), mar_2.keys()).size == 0
    return sum([np.sum(np.absolute(mar_1[k] - mar_2[k])) for k in mar_1.keys()])


def diff_max_marginals(mar_1, mar_2):
    """For each marginal, the diff is calculated according to infinite-norm,
    return summation of all diff

    :param mar_1: dict of first marginal
    :type mar_1: dict
    :param mar_2: dict of the second marginal
    :type mar_2: dict
    :return: distance
    :rtype: float32
    """
    assert np.setdiff1d(mar_1.keys(), mar_2.keys()).size == 0
    return np.max([np.sum(np.absolute(mar_1[k] - mar_2[k]))
                   for k in mar_1.keys()])
