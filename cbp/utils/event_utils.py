import time

import numpy as np


def engine_loop(
        engine_fun,
        max_iter=5000000,
        tolerance=1e-2,
        error_fun=None,
        meassure_fun=None,
        isoutput=False,
        silent=False):
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


def compare_marginals(m1, m2):
  assert not len(np.setdiff1d(m1.keys(), m2.keys()))
  return sum([np.sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])


def diff_max_marginals(m1, m2):
  assert not len(np.setdiff1d(m1.keys(), m2.keys()))
  return np.max([np.sum(np.absolute(m1[k] - m2[k])) for k in m1.keys()])
