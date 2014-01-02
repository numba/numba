import math
import numpy as np


def sum1d(s, e):
    c = 0
    for i in range(s, e):
        c += i
    return c


def sum2d(s, e):
    c = 0
    for i in range(s, e):
        for j in range(s, e):
            c += i * j
    return c


def while_count(s, e):
    i = s
    c = 0
    while i < e:
        c += i
        i += 1
    return c


def copy_arrays(a, b):
    for i in range(a.shape[0]):
        b[i] = a[i]


def copy_arrays2d(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i, j] = a[i, j]


def redefine1():
    x = 0
    for i in range(5):
        x += 1
    x = 0. + x
    for i in range(5):
        x += 1
    return x


def andor(x, y):
    return (x > 0 and x < 10) or (y > 0 and y < 10)


def ifelse1(x, y):
    if x > y:
        return 1
    elif x == 0 or y == 0:
        return 2
    else:
        return 3


def string1(x, y):
    a = "whatzup"
    return a + str(x + y)


def blackscholes_cnd(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val

