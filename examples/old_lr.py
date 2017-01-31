import numba
import numpy as np
import argparse
import time

#parallel = numba.config.NUMBA_NUM_THREADS > 1

#@numba.njit(parallel=parallel)
@numba.njit
def logistic_regression(labels, points, w, iterations):
    # synthetic but deterministic initial values
    points1 = points.transpose()
    for i in range(iterations):
       y = (1.0+np.exp(-labels*(np.dot(w,points))))
       w -= np.dot(((1.0/y-1.0)*labels), points1)
       #y = (1.0/(1.0+np.exp(-labels*(np.dot(w,points))))-1.0) * labels
       #w -= np.dot(y, points1)

    return w

all_tics = []
def tic():
    global all_tics
    all_tics.append(time.time())

def toq():
    global all_tics
    return (time.time() - all_tics.pop())

def main():
    parser = argparse.ArgumentParser(description='Logistic Regression.')
    parser.add_argument('--dimension', dest='dimension', type=int, default=10)
    parser.add_argument('--points', dest='points', type=int, default=20000000)
    parser.add_argument('--iterations', dest='iterations', type=int, default=30)
    args = parser.parse_args()

    np.random.seed(0)
    D = 3
    N = 4
    iterations = 10
    points = np.ones((D, N))
    labels = np.ones((1, N))
    w = 2.0*np.ones((1,D))-1.3
    tic()
    w = logistic_regression(labels, points, w, iterations)
    compiletime = toq()
    print("SELFPRIMED ", compiletime) 
    print("checksum ", w)
 
    D = args.dimension
    N = args.points
    iterations = args.iterations
    print("D=",D," N=",N," iterations=",iterations)
    points = np.random.random((D, N))
    labels = np.random.random((1, N))
    w = 2.0*np.ones((1,D))-1.3
    tic()
    w = logistic_regression(labels, points, w, iterations)
    selftimed = toq()
    print("SELFTIMED ", selftimed) 
    print("checksum: ", np.sum(w))

main()
