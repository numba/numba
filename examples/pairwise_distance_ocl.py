from time import time
import numba
from numba import int32, float32
from math import ceil,sqrt
import numpy as np
import argparse
import timeit

from numba import dppy
from numba.dppy.dppy_driver import driver as ocldrv

parser = argparse.ArgumentParser(description='Program to compute pairwise distance')

parser.add_argument('-n', type=int, required=True, help='Number of points')
parser.add_argument('-d', type=int, default=3, help='Dimensions')

args = parser.parse_args()

@dppy.jit
def pairwise_python(X, D, xshape0, xshape1):
    idx = dppy.get_global_id(0)

    #for i in range(xshape0):
    for j in range(xshape0):
        d = 0.0
        for k in range(xshape1):
            tmp = X[idx, k] - X[j, k]
            d += tmp * tmp
        D[idx, j] = sqrt(d)

def call_ocl():
    global_size = args.n, 1
    local_size = 64, 1, 1
    
    X = np.random.random((args.n, args.d))
    D = np.empty((args.n, args.n))

    #measure running time

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

    start = time()

    dX = device_env.copy_array_to_device(X)
    dD = ocldrv.DeviceArray(device_env.get_env_ptr(), D)

    #pairwise_python[ceil(args.n/1024), 1024](dX, dD, X.shape[0], X.shape[1])
    pairwise_python[device_env,global_size,local_size](dX, dD, X.shape[0], X.shape[1])

    #D = dD.copy_to_host()
    device_env.copy_array_from_device(dD)
    #cuda.synchronize()

    end = time()
    total_time = end - start
    
    print("Total time = " + str(total_time))    

# @cuda.jit('void(float32[:,:], float32[:,:], int32, int32)', target='gpu')
# def pairwise_python(X, D, xshape0, xshape1):

#     idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

#     #for i in range(xshape0):
#     for j in range(xshape0):
#         d = 0.0
#         for k in range(xshape1):
#             tmp = X[idx, k] - X[j, k]
#             d += tmp * tmp
#         D[idx, j] = sqrt(d)

# def call_cuda():
#     X = np.random.random((args.n, args.d))
#     D = np.empty((args.n, args.n))

#     #measure running time

#     start = time()

#     dX = cuda.to_device(X)
#     dD = cuda.to_device(D)

#     pairwise_python[ceil(args.n/1024), 1024](dX, dD, X.shape[0], X.shape[1])

#     D = dD.copy_to_host()
    
#     #cuda.synchronize()

#     end = time()
#     total_time = end - start
    
#     print("Total time = " + str(total_time))

def main():
    call_ocl()
    #call_cuda()
    #call_cuda()
    #call_cuda()

if __name__ == '__main__':
    main()
