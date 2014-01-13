"""
Thanks to Aron Ahmadia
"""

from __future__ import division, print_function

import sys
import math
import numba
from numba import jit, autojit, size_t
import numpy as np
import numpy.testing as npt
try:
    from skimage import img_as_float
except ImportError as e:
    print("skimage not available, skipping")
    sys.exit()

SCALAR_DTYPE = np.float64
# This doesn't work :(
# SCALAR_TYPE  = numba.typeof(SCALAR_DTYPE)
SCALAR_TYPE = numba.float64


def window_floor(idx, radius):
    if radius > idx:
        return 0
    else:
        return idx - radius


def window_ceil(idx, ceil, radius):
    if idx + radius > ceil:
        return ceil
    else:
        return idx + radius


def distance(image, r0, c0, r1, c1):
    d = image[r0, c0, 0] - image[r1, c1, 0]
    s = d * d
    for i in range(1, 3):
        d = image[r0, c0, i] - image[r1, c1, i]
        s += d * d
    return math.sqrt(s)


def pixel_distance(pixel1, pixel2):
    d = pixel1[0] - pixel2[0]
    s = d*d
    for i in range(1, 3):
        d = pixel1[i] - pixel2[i]
        s += d*d
    return math.sqrt(s)


def np_distance(pixel1, pixel2):
    return np.linalg.norm(pixel1-pixel2, 2)


sqrt_3 = math.sqrt(3.0)


def g(d):
    return 1.0 - d/sqrt_3


def np_g(x, y):
    return 1.0 - np_distance(x, y)/sqrt_3


def kernel(image, state, state_next, window_radius):
    changes = 0

    height = image.shape[0]
    width = image.shape[1]

    for j in xrange(width):
        for i in xrange(height):

            winning_colony = state[i, j, 0]
            defense_strength = state[i, j, 1]

            for jj in xrange(window_floor(j, window_radius),
                             window_ceil(j+1, width, window_radius)):
                for ii in xrange(window_floor(i, window_radius),
                                 window_ceil(i+1, height, window_radius)):
                    if (ii == i and jj == j):
                        continue

                    d = image[i, j, 0] - image[ii, jj, 0]
                    s = d * d
                    for k in range(1, 3):
                        d = image[i, j, k] - image[ii, jj, k]
                        s += d * d
                    gval = 1.0 - math.sqrt(s)/sqrt_3

                    attack_strength = gval * state[ii, jj, 1]

                    if attack_strength > defense_strength:
                        defense_strength = attack_strength
                        winning_colony = state[ii, jj, 0]
                        changes += 1

            state_next[i, j, 0] = winning_colony
            state_next[i, j, 1] = defense_strength

    return changes


def growcut(image, state, max_iter=20, window_size=3):
    """Grow-cut segmentation (Numba accelerated).

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.
    state : (M, N, 2) ndarray
        Initial state, which stores (foreground/background, strength) for
        each pixel position or automaton.  The strength represents the
        certainty of the state (e.g., 1 is a hard seed value that remains
        constant throughout segmentation).
    max_iter : int, optional
        The maximum number of automata iterations to allow.  The segmentation
        may complete earlier if the state no longer varies.
    window_size : int, optional
        Size of the neighborhood window.

    Returns
    -------
    mask : ndarray
        Segmented image.  A value of zero indicates background, one foreground.

    """

    image = img_as_float(image)

    window_radius = (window_size - 1) // 2

    changes = 1
    n = 0

    state_next = np.empty_like(state)

    while changes > 0 and n < max_iter:
        changes = 0
        n += 1
        changes = kernel(image, state, state_next, window_radius)
        state_next, state = state, state_next
        #print n, changes
        print('.', end='')
    print('')
    return state_next[:, :, 0]


def create_numba_funcs(scalar_type=SCALAR_TYPE):
    this = sys.modules[__name__]

    pixel_type = scalar_type[:]
    image_type = scalar_type[:, :, :]
    state_type = scalar_type[:, :, :]

    this._numba_window_floor = jit(nopython=True,
                                   argtypes=[size_t, size_t],
                                   restype=size_t)(_py_window_floor)

    this._numba_window_ceil = jit(nopython=True,
                                  argtypes=[size_t, size_t, size_t],
                                  restype=size_t)(_py_window_ceil)

    this._numba_distance = jit(nopython=True,
                               argtypes=[image_type,
                                         size_t, size_t, size_t, size_t],
                               restype=scalar_type)(_py_distance)

    this._numba_np_distance = jit(nopython=False,
                                  argtypes=[pixel_type, pixel_type],
                                  restype=scalar_type)(_py_np_distance)

    this._numba_g = jit(nopython=True,
                        argtypes=[scalar_type],
                        restype=scalar_type)(_py_g)

    this._numba_np_g = jit(nopython=False,
                           argtypes=[pixel_type, pixel_type],
                           restype=scalar_type)(_py_np_g)

    this._numba_kernel = autojit(nopython=True)(_py_kernel)
    # the below code does not work
    # this._numba_kernel        = jit(nopython=False,
    #                                  argtypes=[image_type,
    #                                            state_type,
    #                                            state_type,
    #                                            size_t],
    #                                  restype=int_,
    #                                  attack_strength=scalar_type,
    #                                  defense_strength=scalar_type,
    #                                  winning_colony=scalar_type)(_py_kernel)


def debug():
    this = sys.modules[__name__]
    this.window_floor = _py_window_floor
    this.window_ceil = _py_window_ceil
    this.distance = _py_distance
    this.np_distance = _py_np_distance
    this.g = _py_g
    this.np_g = _py_np_g
    this.kernel = _py_kernel


def optimize():
    this = sys.modules[__name__]
    this.window_floor = _numba_window_floor
    this.window_ceil = _numba_window_ceil
    this.distance = _numba_distance
    this.np_distance = _numba_np_distance
    this.g = _numba_g
    this.np_g = _numba_np_g
    this.kernel = _numba_kernel


# protected Pythonic versions of code:
_py_window_floor = window_floor
_py_window_ceil = window_ceil
_py_distance = distance
_py_np_distance = np_distance
_py_g = g
_py_np_g = np_g
_py_kernel = kernel


def test_window_floor_ceil():

    assert 3 == window_floor(4, 1)
    assert 0 == window_floor(1, 4)

    assert 3 == window_ceil(3, 3, 1)
    assert 5 == window_ceil(4, 5, 1)


def test_distance():
    image = np.zeros((2, 2, 3), dtype=SCALAR_DTYPE)
    image[0, 1] = [1, 1, 1]
    image[1, 0] = [0.5, 0.5, 0.5]

    assert 0.0 == distance(image, 0, 0, 0, 0)
    assert abs(math.sqrt(3) - distance(image, 0, 0, 0, 1)) < 1e-15
    assert abs(math.sqrt(3/4) - distance(image, 0, 1, 1, 0)) < 1e-15

    pixel1 = np.asarray([0.0, 0.0, 0.0], dtype=SCALAR_DTYPE)
    pixel2 = np.asarray([1.0, 1.0, 1.0], dtype=SCALAR_DTYPE)
    pixel3 = np.asarray([0.5, 0.5, 0.5], dtype=SCALAR_DTYPE)

    assert 0.0 == np_distance(pixel1, pixel1)
    assert abs(math.sqrt(3) - np_distance(pixel1, pixel2)) < 1e-15
    assert abs(math.sqrt(3/4) - np_distance(pixel2, pixel3)) < 1e-15


def test_g():
    image = np.zeros((2, 2, 3), dtype=SCALAR_DTYPE)
    image[0, 1] = [1, 1, 1]
    image[1, 0] = [0.5, 0.5, 0.5]

    assert 1.0 == g(distance(image, 0, 0, 0, 0))
    assert abs(0 - g(distance(image, 0, 0, 0, 1))) < 1e-15
    assert abs(0.5 - g(distance(image, 0, 1, 1, 0))) < 1e-15

    pixel1 = np.asarray([0.0, 0.0, 0.0], dtype=SCALAR_DTYPE)
    pixel2 = np.asarray([1.0, 1.0, 1.0], dtype=SCALAR_DTYPE)
    pixel3 = np.asarray([0.5, 0.5, 0.5], dtype=SCALAR_DTYPE)

    assert 1.0 == np_g(pixel1, pixel1)
    assert abs(0 - np_g(pixel1, pixel2)) < 1e-15
    assert abs(0.5 - np_g(pixel2, pixel3)) < 1e-15


def test_kernel():
    image = np.zeros((3, 3, 3), dtype=SCALAR_DTYPE)
    state = np.zeros((3, 3, 2), dtype=SCALAR_DTYPE)
    state_next = np.empty_like(state)

    # colony 1 is strength 1 at position 0,0
    # colony 0 is strength 0 at all other positions
    state[0, 0, 0] = 1
    state[0, 0, 1] = 1

    # window_size 1, colony 1 should propagate to three neighbors
    changes = kernel(image, state, state_next, 1)
    assert(3 == changes)
    npt.assert_array_equal(state_next[0:2, 0:2], 1)
    npt.assert_array_equal(state_next[2, :], 0)
    npt.assert_array_equal(state_next[2, :], 0)

    # window_size 1, colony 1 should propagate to entire image
    changes = kernel(image, state, state_next, 2)
    assert(8 == changes)
    npt.assert_array_equal(state_next, 1)


def test():
    test_window_floor_ceil()
    test_distance()
    test_g()
    test_kernel()


# create numba versions of code
create_numba_funcs()


if __name__ == "__main__":
    # always verify pure Python code first
    test()
    # then test optimized variants
    optimize()
    test()


# replace default function calls with numba calls
optimize()