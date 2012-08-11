from numba import d
from numba.decorators import jit as jit

def sum(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result

csum = jit(ret_type=d, arg_types=[d[:,:]])(sum)

