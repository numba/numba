# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Linear Regression with Gradient Descent Algorithm
# 
# This notebook demonstrates the implementation of linear regression with gradient descent algorithm.  
# 
# Consider the following implementation of the gradient descent loop with NumPy arrays based upon [1]:

# <codecell>

def gradient_descent_numpy(X, Y, theta, alpha, num_iters):
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

# <markdowncell>

# To speedup this implementation with Numba, we need to add the `@jit` decorator to annotate the function signature.  Then, we need to expand the NumPy array expressions into a loop.  The resulting code is shown below:

# <codecell>

from numba import autojit, jit, f8, int32, void

@jit(void(f8[:], f8[:], f8[:], f8, int32))
def gradient_descent_numba(X, Y, theta, alpha, num_iters):
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

# <markdowncell>

# The rest of the code generates some artificial data to test our linear regression algorithm.

# <codecell>

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

    print "x-offset = {}    slope = {}".format(*theta)
    print "time elapsed: {} s".format(timing)

    return theta, timing


def plot(X, theta, c='r'):
    result = theta[0] + theta[1] * X
    pylab.plot(X, result, c=c, linewidth=2)

# <markdowncell>

# We will a benchmark with 50 elements to compare the pure python version against the numba version.

# <codecell>

N = 50
X, Y = populate_data(N)
pylab.scatter(X, Y, marker='o', c='b')
pylab.title('Linear Regression')

print 'Python'.center(30, '-')
theta_python, time_python = run(gradient_descent_numpy, X, Y)

print 'Numba'.center(30, '-')
theta_numba, time_numba  = run(gradient_descent_numba, X, Y)

# make sure all method yields the same result
assert np.allclose(theta_python, theta_numba)

print 'Summary'.center(30, '=')
print 'Numba speedup %.1fx' % (time_python / time_numba)

plot(X, theta_numba, c='r')
pylab.show()

# <markdowncell>

# 
# ## References
# 
# [1] http://aimotion.blogspot.com/2011/10/machine-learning-with-python-linear.html

