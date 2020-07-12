import numpy as np
from numba import njit


def nd_multiexpand(input_data, target_shape, which_dims):
    init_shape = [1] * len(target_shape)
    expand_shape = list(target_shape)
    for i, cur_dim in enumerate(which_dims):
        assert input_data.shape[i] == target_shape[cur_dim]
        init_shape[cur_dim] = target_shape[cur_dim]
        expand_shape[cur_dim] = 1

    out = input_data.reshape(tuple(init_shape))
    return np.tile(out, tuple(expand_shape))


@njit
def nd_expand(inputdata, target_shape, expand_dim):
    """expand ndarray to target shape

    :param inputdata:
    :type inputdata: list or 1d ndarray
    :param target_shape:
    :type target_shape: tuple
    :param expand_dim: [description]
    :type expand_dim: int
    :return: expanded ndarray with `target_shape`
    :rtype: ndarray

    >>> inputdata = [1,2]
    >>> target_shape = (1,2,3)
    >>> expand_dim = 1
    >>> output = np.array([
                [1,1,1],
                [2,2,2]
            ])
    """
    assert inputdata.ndim == 1
    rtn = np.zeros(target_shape)
    for idx in np.ndindex(target_shape):
        rtn[idx] = inputdata[idx[expand_dim]]
    return rtn


@njit
def reduction_ndarray(ndarray, reduction_index):
    """reduct ndarray according to one index

    :param ndarray: [description]
    :type ndarray: ndarray
    :param reduction_index: [description]
    :type reduction_index: int
    :return: [description]
    :rtype: ndarray
    """
    rtn = np.zeros(ndarray.shape[reduction_index])
    for idx, value in np.ndenumerate(ndarray):
        rtn[idx[reduction_index]] += value
    return rtn


@njit
def batch_normal_angle(angle):
    delta_x = np.cos(angle)
    delta_y = np.sin(angle)
    return np.arctan2(delta_y, delta_x)


def empirical_marginal(traj, num_bins):
    marginal = []
    for i in range(traj.shape[1]):
        bins, _ = np.histogram(
            traj[:, i], np.arange(num_bins + 1))
        marginal.append(bins / np.sum(bins))

    return np.array(marginal).reshape(traj.shape[1], num_bins)
