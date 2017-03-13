import numba
import numpy as np
import argparse
import time

parallel = numba.config.NUMBA_NUM_THREADS > 1

@numba.njit(parallel=parallel)
def logistic_regression(Y,X,w,iterations):
    for i in range(iterations):
        w -= np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X,w))) - 1.0) * Y),X)
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
    points = np.ones((N, D))
    labels = np.ones(N)
    w = 2.0*np.ones(D)-1.3
    tic()
    w = logistic_regression(labels, points, w, iterations)
    compiletime = toq()
    print("SELFPRIMED ", compiletime)
    print("checksum ", w)

    D = args.dimension
    N = args.points
    iterations = args.iterations
    print("D=",D," N=",N," iterations=",iterations)
    points = np.random.random((N, D))
    labels = np.random.random(N)
    w = 2.0*np.ones(D)-1.3
    tic()
    w = logistic_regression(labels, points, w, iterations)
    selftimed = toq()
    print("SELFTIMED ", selftimed)
    print("checksum: ", np.sum(w))

main()
