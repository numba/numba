# -*- coding: utf-8 -*-
'''
Only added decorators to the linregr_python.py implementation.
'''
from __future__ import print_function, division, absolute_import

import numbapro
from numba import autojit, jit, f8, int32, void

@jit(void(f8[:], f8[:], f8[:], f8, int32))
def gradient_descent(X, Y, theta, alpha, num_iters):
    m = Y.shape[0]

    theta_x = 0.0
    theta_y = 0.0

    for i in range(num_iters):
        predict = theta_x + theta_y * X
        err_x = (predict - Y)
        err_y = (predict - Y) * X
        theta_x = theta_x - alpha * (1.0 / m) * err_x.sum()
        theta_y = theta_y - alpha * (1.0 / m) * err_y.sum()

    theta[0] = theta_x
    theta[1] = theta_y


