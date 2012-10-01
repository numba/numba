from numba import double
from numba.decorators import jit as jit

def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

csum2d = jit(restype=double, argtypes=[double[:,:]])(sum2d)

from numpy import random
arr = random.randn(100,100)

import time
start = time.time()
res = sum2d(arr)
duration = time.time() - start
print "Result from python is %s in %s (msec)" % (res, duration*1000)

start = time.time()
res = csum2d(arr)
duration2 = time.time() - start
print "Result from compiled is %s in %s (msec)" % (res, duration2*1000)

print "Speed up is %s" % (duration / duration2)
