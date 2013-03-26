# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import linregr_python, linregr_numba, linregr_numbapro
import numpy as np
import pylab
from timeit import default_timer as timer

def populate_data(N):
    noise = np.random.random(N).astype(np.float64)
    X = np.arange(N, dtype=np.float64)
    slope = 3
    Y = noise * (slope * X)
    return X, Y

def run(gradient_descent, X, Y, iterations=1000, alpha=1e-6):
    theta = np.empty(2, dtype=X.dtype)

    ts = timer()
    gradient_descent(X, Y, theta, alpha, iterations)
    te = timer()

    timing = te - ts

    print("x-offset = {}    slope = {}".format(*theta))
    print("time elapsed: {} s".format(timing))

    return theta, timing


def plot(X, theta, c='r'):
    result = theta[0] + theta[1] * X
    pylab.plot(X, result, c=c, linewidth=2)


def main():
    N = 50
    X, Y = populate_data(N)
    pylab.scatter(X, Y, marker='o', c='b')
    pylab.title('Linear Regression')

    print('Python'.center(80, '-'))
    theta_python, time_python = run(linregr_python.gradient_descent, X, Y)

    print('Numba'.center(80, '-'))
    theta_numba, time_numba  = run(linregr_numba.gradient_descent, X, Y)

    print('NumbaPro'.center(80, '-'))
    theta_numbapro, time_numbapro  = run(linregr_numbapro.gradient_descent, X, Y)

    # make sure all method yields the same result
    assert np.allclose(theta_python, theta_numba)
    assert np.allclose(theta_python, theta_numbapro)

    print('Summary'.center(80, '='))
    print('Numba speedup %.1fx' % (time_python / time_numba))
    print('NumbaPro speedup %.1fx' % (time_python / time_numbapro))

    plot(X, theta_numba, c='r')
    pylab.show()

if __name__ == '__main__':
    main()

