import numpy as np


def expand_ndarray(inputdata, target_shape, expand_dim):
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
    assert target_shape[expand_dim] == len(inputdata)
    init_shape = [1 for i in range(len(target_shape))]
    init_shape[expand_dim] = target_shape[expand_dim]
    out = np.array(inputdata).reshape(tuple(init_shape))

    outshape_list = list(target_shape)
    outshape_list[expand_dim] = 1

    outshape = tuple(outshape_list)
    return np.tile(out, outshape)


def reduction_ndarray(ndarray, reduction_index):
    """reduct ndarray according to one index

    :param ndarray: [description]
    :type ndarray: ndarray
    :param reduction_index: [description]
    :type reduction_index: int
    :return: [description]
    :rtype: ndarray
    """
    sum_dims = [j for j in range(ndarray.ndim) if not j == reduction_index]
    sum_dims.sort(reverse=True)
    collapsing_marginal = ndarray
    for cur_dim in sum_dims:
        collapsing_marginal = collapsing_marginal.sum(cur_dim)  # lose 1 dim

    return collapsing_marginal


def ndarray_denominator(ndarray):
    check_index = np.isclose(ndarray, 0)
    if check_index.any():
        print("used")
        ndarray[check_index] = np.inf
    return ndarray


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
