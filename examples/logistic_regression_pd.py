import sys
import time
import numpy as np
import os
import numba

@numba.njit
def logistic_regression(X,Y,w,iterations):
    for i in range(iterations):
        w -= ((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y * X.T).sum(1)
    return w
    

if __name__ == "__main__":

    D = 10  # Number of dimensions
    iterations = 20
    N = 10
    if len(sys.argv)>1:
        N = int(sys.argv[1])
    if len(sys.argv)>2:
        iterations = int(sys.argv[2])
    if len(sys.argv)>3:
        D = int(sys.argv[3])
    print("N %d D %d iterations %d" %(N,D,iterations))

    # Initialize w to a random value
    X = np.random.ranf(size=(N,D))
    Y = np.random.ranf(size=(N))

    start = time.time()

    w = 2 * np.random.ranf(size=D) - 1
    print("Initial w: " + str(w))
    logistic_regression(X,Y,w,iterations)

    print("Final w: " + str(w))
    print("lr exec time %f" % (time.time()-start))
