#! /usr/bin/env python
from __future__ import print_function
from timeit import default_timer as time

import sys
import argparse
import time
import math
import numpy as np
from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv

@dppy.kernel
def calculate_distance(A, centroids, dist, num_centers):
    i = dppy.get_global_id(0)
    for j in range(num_centers):
        s = 0
        for idx in range(A.shape[1]):
            s += (A[i, idx] - centroids[j, idx]) * (A[i, idx] - centroids[j, idx])
        dist[i, j] = math.sqrt(s)

@dppy.kernel
def calculate_labels(labels, dist):
    i = dppy.get_global_id(0)
    labels[i] = dist[i,:].argmin()

@dppy.kernel
def update_centroids(A, centroids, labels, num_features):
    i = dppy.get_global_id(0)
    for j in range(num_features):
        r = A[labels==i, j]
        s1 = 0
        for item in r:
            s1 += item
        r = labels==i
        s2 = 0
        for item in r:
            s2 += item
        centroids [i, j] = s1/s2


def main():
    parser = argparse.ArgumentParser(description='K-Means NUMBA-PVC')
    #parser.add_argument('--size', dest='size', type=int, default=1000000)
    parser.add_argument('--size', dest='size', type=int, default=100)
    parser.add_argument('--features', dest='features', type=int, default=10)
    parser.add_argument('--centers', dest='centers', type=int, default=5)
    parser.add_argument('--iterations', dest='iterations', type=int, default=20)
    args = parser.parse_args()

    size = args.size
    features = args.features
    centers = args.centers
    iterations = args.iterations

    # Select a device for executing the kernel
    device_env = None
    try:
        device_env = ocldrv.runtime.get_gpu_device()
        print("Selected GPU device")
    except:
        try:
            device_env = ocldrv.runtime.get_cpu_device()
            print("Selected CPU device")
        except:
            print("No OpenCL devices found on the system")
            raise SystemExit()

    #np.random.seed(0)
    A = np.random.ranf((size, features))
    centroids = np.random.ranf((centers, features))
    dist = np.random.ranf((size, centers))
    labels = np.random.ranf(size)

    dA = device_env.copy_array_to_device(A)
    dCentroids = device_env.copy_array_to_device(centroids)
    dDist = device_env.copy_array_to_device(dist)
    dLabels = device_env.copy_array_to_device(labels)

    t1 = time.time()
    for l in range(iterations):
        global_size = size
        calculate_distance[device_env, global_size](dA, dCentroids, dDist, centers)
        calculate_labels[device_env, global_size](dLabels, dDist)

        global_size = centers
        update_centroids[device_env, global_size](dA, dCentroids, dLabels, features)

    t = time.time()-t1
    device_env.copy_array_from_device(dCentroids)
    print(t)

if __name__ == '__main__':
    main()
