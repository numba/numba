'''
Copyright (c) 2015, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:
- Redistributions of source code must retain the above copyright notice, 
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, 
  this list of conditions and the following disclaimer in the documentation 
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
THE POSSIBILITY OF SUCH DAMAGE.
'''

import numba
import numpy as np
import argparse
import time
import math

#parallel = numba.config.NUMBA_NUM_THREADS > 1
parallel = False

@numba.njit
def argmin(dist):
  N, numCenter = dist.shape
  mins = np.empty((N,), dtype=np.int64)
  for i in range(N):
    k = 0
    m = dist[i, 0]
    for j in range(numCenter): 
      if dist[i, j] < m:
        k = j
        m = dist[i, j]
    mins[i] = k
  return mins 


@numba.njit(parallel=parallel)
def kmeans(numCenter, iterNum, points):
    N, D = points.shape # number of features, instances
    centroids = np.random.rand(numCenter, D)

    for l in range(iterNum):
        # dist :: Array{Array{Float64,1},1} = [ Float64[sqrt(sum((points[:,i].-centroids[:,j]).^2)) for j in 1:numCenter] for i in 1:N]
        dist = np.empty((N, numCenter))
        for i in range(N):
            for j in range(numCenter):
                dist[i, j] = math.sqrt(np.sum(np.power(points[i,:] - centroids[j,:], 2.0)))
        # labels :: Array{Int, 1} = [indmin(dist[i]) for i in 1:N]
        labels = argmin(dist)
        # labels = np.argmin(dist, 1) 
        # labels = dist.argmin(1)
        # centroids :: Array{Float64,2} = [ sum(points[j,labels.==i])/sum(labels.==i) for j in 1:D, i in 1:numCenter]
        for i in range(numCenter):
          m = np.sum(labels == i)
          for j in range(D):
            centroids[i,j] = np.sum(points[labels == i, j]) / m
    return centroids

all_tics = []
def tic():
    global all_tics
    all_tics.append(time.time())

def toq():
    global all_tics
    return (time.time() - all_tics.pop())

def main():
    parser = argparse.ArgumentParser(description='k-means clustering algorithm')
    parser.add_argument('--iterations', dest='iterations', type=int, default=30)
    parser.add_argument('--size', dest='size', type=int, default=50000)
    parser.add_argument('--centers', dest='centers', type=int, default=5)
    args = parser.parse_args()
    iterations = args.iterations
    size = args.size
    numCenter = args.centers

    D = 20
    np.random.seed(0)

    print("iterations = ", iterations)
    print("centers= ", numCenter) 
    print("number of points = ", size) 
    points = np.random.rand(size, D)

    tic()
    kmeans(numCenter, 2, np.random.rand(D,100))
    time = toq()
    print("SELFPRIMED ", time)

    tic()
    centroids_out = kmeans(numCenter, iterations, points)
    time = toq()
    print("result = ", centroids_out)
    print("rate = ", iterations / time, " iterations/sec")
    print("SELFTIMED ", time)

main()
