#
# Copyright (c) 2017 Intel Corporation
# SPDX-License-Identifier: BSD-2-Clause
#

from numba import njit
import numpy as np
from math import sqrt
import argparse
import time

def kmeans(A, numCenter, numIter, N, D, init_centroids):
    centroids = init_centroids

    for l in range(numIter):
        dist = np.array([[sqrt(np.sum((A[i,:]-centroids[j,:])**2))
                                for j in range(numCenter)] for i in range(N)])
        labels = np.array([dist[i,:].argmin() for i in range(N)])

        centroids = np.array([[np.sum(A[labels==i, j])/np.sum(labels==i)
                                 for j in range(D)] for i in range(numCenter)])

    return centroids

def main():
    parser = argparse.ArgumentParser(description='K-Means')
    parser.add_argument('--size', dest='size', type=int, default=1000000)
    parser.add_argument('--features', dest='features', type=int, default=10)
    parser.add_argument('--centers', dest='centers', type=int, default=5)
    parser.add_argument('--iterations', dest='iterations', type=int, default=20)
    args = parser.parse_args()
    size = args.size
    features = args.features
    centers = args.centers
    iterations = args.iterations

    np.random.seed(0)
    init_centroids = np.random.ranf((centers, features))
    kmeans(np.random.ranf((3000, features)), centers, 1, 3000, features, init_centroids)
    print("size:", size)
    A = np.random.ranf((size, features))

    t1 = time.time()
    res = kmeans(A, centers, iterations, size, features, init_centroids)
    t = time.time()-t1
    print("checksum:", res.sum())
    print("SELFTIMED:", t)

if __name__ == '__main__':
    main()
