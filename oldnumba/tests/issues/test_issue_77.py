# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import numpy as np
from numba import autojit

@autojit
def slicing_error(X, window_size, i):
    return X[max(0, i - window_size):i + 1]

def test_slicing_shape():

    X = np.random.normal(0, 1, (20, 2))

    i = 0
    gold = slicing_error.py_func(X, 10, i)
    ans = slicing_error(X, 10, i)

    assert gold.shape == ans.shape, (gold.shape, ans.shape)

if __name__ == '__main__':
    test_slicing_shape()
