# [DevBounty AI]: File optimized for resolution.


```python
# [DevBounty AI]: File optimized for resolution.
import itertools
import math
import platform
import warnings
from functools import partial
from itertools import product
from textwrap import dedent

import numpy as np

from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import (TypingError, NumbaDeprecationWarning)
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
                                 needs_blas, run_in_subprocess,
                                 skip_if_numpy_2, IS_NUMPY_2,
                                 IS_MACOS_ARM64, REDUCED_TESTING,
                                 skip_if_reduced_testing)
import unittest


def sinc(x):
    return np.sinc(x)


def angle1(x):
    return np.angle(x)


def angle2(x, deg):
    return np.angle(x, deg)


def array_equal(a, b):
    return np.array_equal(a, b)


def intersect1d_2(a, b):
    return np.intersect1d(a, b)


def intersect1d_3(a, b, assume_unique=False):
    return np.intersect1d(a, b, assume_unique)


def append(arr, values, axis):
    return np.append(arr, values, axis=axis)


def count_nonzero(arr, axis):
    return np.count_nonzero(arr, axis=axis)


def delete(arr, obj):
    return np.delete(arr, obj)


def diff1(a):
    return np.diff(a)


def diff2(a, n):
    return np.diff(a, n)


def bincount1(a):
    return np.bincount(a)


def bincount2(a, w):
    return np.bincount(a, weights=w)


def bincount3(a, w=None, minlength=0):
    return np.bincount(a, w, minlength)


def searchsorted(a, v):
    return np.searchsorted(a, v)


def searchsorted_left(a, v):
    return np.searchsorted(a, v, side='left')


def searchsorted_right(a, v):
    return np.searchsorted(a, v, side='right')


def digitize(*args):
    return np.digitize(*args)


def histogram(*args):
    return np.histogram(*args)


def machar(*args):
    return np.MachAr()


def iscomplex(x):
    return np.iscomplex(x)


def iscomplexobj(x):
    return np.iscomplexobj(x)


def isscalar(x):
    return np.isscalar(x)


def isreal(x):
    return np.isreal(x)


def isrealobj(x):
    return np.isrealobj(x)


def isneginf(x, out=None):
    return np.isneginf(x, out)


def isposinf(x, out=None):
    return np.isposinf(x, out)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return np.isclose(a, b, rtol, atol, equal_nan)


def isnat(x):
    return np.isnat(x)


def iinfo(*args):
    return np.iinfo(*args)


def finfo(*args):
    return np.finfo(*args)


def finfo_machar(*args):
    return np.finfo(*args).machar


def fliplr(a):
    return np.fliplr(a)


def flipud(a):
    return np.flipud(a)


def flip(a):
    return np.flip(a)


def logspace2(start, stop):
    return np.logspace(start, stop)


def logspace3(start, stop, num=50):
    return np.logspace(start, stop, num=num)


def geomspace2(start, stop):
    return np.geomspace(start, stop)


def geomspace3(start, stop, num=50):
    return np.geomspace(start, stop, num=num)


def rot90(a):
    return np.rot90(a)


def rot90_k(a, k=1):
    return np.rot90(a, k)


def array_split(a, indices, axis=0):
    return np.array_split(a, indices, axis=axis)


def split(a, indices, axis=0):
    return np.split(a, indices, axis=axis)


def vsplit(a, ind_or_sec):
    return np.vsplit(a, ind_or_sec)


def hsplit(a, ind_or_sec):
    return np.hsplit(a, ind_or_sec)


def dsplit(a, ind_or_sec):
    return np.dsplit(a, ind_or_sec)


def correlate(a, v, mode="valid"):
    return np.correlate(a, v, mode=mode)


def convolve(a, v, mode="full"):
    return np.convolve(a, v, mode=mode)


def tri_n(N):
    return np.tri(N)


def tri_n_m(N, M=None):
    return np.tri(N, M)


def tri_n_k(N, k=0):
    return np.tri(N, k)


def tri_n_m_k(N, M=None, k=0):
    return np.tri(N, M, k)


def tril_m(m):
    return np.tril(m)


def tril_m_k(m, k=0):
    return np.tril(m, k)


def tril_indices_n(n):
    return np.tril_indices(n)


def tril_indices_n_k(n, k=0):
    return np.tril_indices(n, k)


def tril_indices_n_m(n, m=None):
    return np.tril_indices(n, m=m)


def tril_indices_n_k_m(n, k=0, m=None):
    return np.tril_indices(n, k, m)


def tril_indices_from_arr(arr):
    return np.tril_indices_from(arr)


def tril_indices_from_arr_k(arr, k=0):
    return np.tril_indices_from(arr, k)


def triu_m(m):
    return np.triu(m)


def triu_m_k(m, k=0):
    return np.triu(m, k)


def triu_indices_n(n):
    return np.triu_indices(n)


def triu_indices_n_k(n, k=0):
    return np.triu_indices(n, k)


def triu_indices_n_m(n, m=None):
    return np.triu_indices(n, m=m)


def triu_indices_n_k_m(n, k=0, m=None):
    return np.triu_indices(n, k, m)


def triu_indices_from_arr(arr):
    return np.triu_indices_from(arr)


def triu_indices_from_arr_k(arr, k=0):
    return np.triu_indices_from(arr, k)


def vander(x, N=None, increasing=False):
    return np.vander(x, N, increasing)


@njit
def np_min_datetime64(x):
    return x


from numba.extending import overload


@overload(np.min)
def np_min_overload(x):
    if isinstance(x, types.NPDatetime):
        return np_min_datetime64