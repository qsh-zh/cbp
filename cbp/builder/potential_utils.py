import numpy as np
from scipy import signal


def diagonal_potential(d_1: int, d_2: int,
                       rng: np.random.RandomState) -> np.ndarray:
    factor_potential = rng.randint(4, 6, size=(d_1, d_2)) * 1.0
    dim = np.min([d_1, d_2])
    identity = np.eye(dim)
    if rng.normal(size=1) > 1:
        identity = np.flip(identity, axis=0)
    if d_2 > d_1:
        diagonal = np.concatenate([identity, np.zeros(dim, d_2 - dim)], axis=1)
    elif d_2 < d_1:
        diagonal = np.concatenate([identity, np.zeros(d_1 - dim, dim)], axis=0)
    else:
        diagonal = identity

    #diagonal = rng.permutation(diagonal)
    diagonal_dominance = np.exp(factor_potential) + diagonal * 50000
    return diagonal_dominance / np.mean(diagonal_dominance)


def diagonal_potential_different(d_1: int, d_2: int,
                                 rng: np.random.RandomState) -> np.ndarray:
    factor_potential = rng.randint(4, 10, size=(d_1, d_2)) * 1.0
    dim = np.min([d_1, d_2])
    identity = np.eye(dim)
    if rng.normal(size=1) > 1:
        identity = np.flip(identity, axis=0)
    if d_2 > d_1:
        diagonal = np.concatenate([identity, np.zeros(dim, d_2 - dim)], axis=1)
    elif d_2 < d_1:
        diagonal = np.concatenate([identity, np.zeros(d_1 - dim, dim)], axis=0)
    else:
        diagonal = identity

    #diagonal = rng.permutation(diagonal)
    diagonal_dominance = np.exp(factor_potential) + diagonal * 50000
    return diagonal_dominance / np.mean(diagonal_dominance)


def diagonal_potential_conv(d_1: int, d_2: int,
                            rng: np.random.RandomState) -> np.ndarray:
    kernel = np.array([[0.5, 1, 0.5]])
    factor_potential = rng.randint(4, 10, size=(d_1, d_2)) * 1.0
    dim = np.min([d_1, d_2])
    identity = np.eye(dim)
    if rng.normal(size=1) > 1:
        identity = np.flip(identity, axis=0)
    if d_2 > d_1:
        diagonal = np.concatenate([identity, np.zeros(dim, d_2 - dim)], axis=1)
    elif d_2 < d_1:
        diagonal = np.concatenate([identity, np.zeros(d_1 - dim, dim)], axis=0)
    else:
        diagonal = identity

    diagonal = rng.permutation(diagonal)
    diagonal_dominance = np.exp(factor_potential) + diagonal * 50000
    diagonal_dominance /= np.mean(diagonal_dominance)

    # return diagonal_dominance / np.mean(diagonal_dominance)
    return signal.convolve2d(diagonal_dominance, kernel, mode="same")


def identity_potential(d_1: int, d_2: int,
                       rng: np.random.RandomState) -> np.ndarray:  # pylint: disable=unused-argument
    dim = np.min([d_1, d_2])
    identity = np.eye(dim)
    if d_2 > d_1:
        diagonal = np.concatenate([identity, np.zeros(dim, d_2 - dim)], axis=1)
    elif d_2 < d_1:
        diagonal = np.concatenate([identity, np.zeros(d_1 - dim, dim)], axis=0)
    else:
        diagonal = identity

    return diagonal
