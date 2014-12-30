# -*- coding: utf-8 -*-
'''
Numba does not support array expressions.
Expand the array-expression into loops.
'''
from __future__ import print_function, division, absolute_import
from numba import autojit, jit, f8, int32, void

@jit(void(f8[:], f8[:], f8[:], f8, int32))
def gradient_descent(X, Y, theta, alpha, num_iters):
    m = Y.shape[0]

    theta_x = 0.0
    theta_y = 0.0

    for i in range(num_iters):
        err_acc_x = 0.0
        err_acc_y = 0.0
        for j in range(X.shape[0]):
            predict = theta_x + theta_y * X[j]
            err_acc_x += predict - Y[j]
            err_acc_y += (predict - Y[j]) * X[j]
        theta_x = theta_x - alpha * (1.0 / m) * err_acc_x
        theta_y = theta_y - alpha * (1.0 / m) * err_acc_y

    theta[0] = theta_x
    theta[1] = theta_y

