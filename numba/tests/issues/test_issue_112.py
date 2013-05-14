# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np

from numba import autojit, register_callable, npy_intp, typesystem

restype = typesystem.tuple_(npy_intp[:, :], 2)

@register_callable(restype(npy_intp[:], npy_intp[:]))
def meshgrid(x, y):
    return np.meshgrid(x, y)

@autojit
def run_meshgrid(size):
    x1d = np.arange(-size, size + 1)
    y1d = np.arange(-size, size + 1)
    x, y = meshgrid(x1d, y1d)
    return x, y

if __name__ == '__main__':
    size = 3
    nb_gauss = run_meshgrid(size)
