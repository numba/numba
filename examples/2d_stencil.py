'''
This script executes 2D Stencil operation on images in grayscale.
It will calculate LBP Texture of the Image and Histogram of the LBP Values.
Usage:
Run without argument will use builtin image:
    python 2d_stencil.py
= Getting The Requirements =
For Conda user, run the following to ensure the dependencies are fulfilled:
    conda install scipy matplotlib
'''

from __future__ import print_function
import numpy as np
import sys
from timeit import default_timer as timer
from scipy import *
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from numba import cuda, void, int32, jit
from scipy import misc, ndimage
import math

# I distribute kernel code in two Phaces.
# 1. ________Copy data to shared memory_______:
#       for 256*256 screen I am having 32*32 threads per block
#       Totaling 8*8 blocks
#       But Operations Will be applied on 30*30 window
#       Have to add apron for stensil function
# 2. _______ Apply the stencil function over the shared data.



# Configuration for 256*256 window
n = 256
threadCount = 32

radius = 1
BIN_COUNT = 255

@jit([void(int32[:,:], int32[:])], target='cuda')
def lbp_texture(arry, hist):

    # We have 32*32 threads per block
    A = cuda.shared.array(shape=(32,32), dtype=int32)

    # H = cuda.shared.array(BIN_COUNT, dtype=int32)
    x,y = cuda.grid(2)

    ty = cuda.threadIdx.x
    tx = cuda.threadIdx.y

    A[ty,tx] = arry[x,y]


    cuda.syncthreads()

    threadCountX = A.shape[0] - 1
    threadCountY = A.shape[1] - 1
    # If within x range and y range then calculate the LBP discriptor along
    # with histogram value to specific bin

    # Other wise Ignore the Value
    if (ty > 0 and  (threadCountX-ty) > 0 ) and (tx > 0 and (threadCountY-tx) > 0):
    #     # You can do the Processing here. ^_^
        code = 0
        #  We need to make sure that each value is accessable to each thread
        center = A[ty, tx]

        # Compiler optimization: By loop unrolling
        # turns out twice faster than rolled version for over
        # 16*16 window
        code |= (1 if A[ty-1][tx-1] > center else 0 ) <<  7
        code |= (1 if A[ty][tx-1] > center else 0)  << 6
        code |= (1 if A[ty+1][tx-1] > center else 0 )<< 5
        code |= (1 if A[ty+1][tx] > center else 0 ) << 4
        code |= (1 if A[ty+1][tx+1] > center else 0 ) << 3
        code |= (1 if A[ty][tx+1] > center else 0 ) << 2
        code |= (1 if A[ty-1][tx+1] > center else 0 )<< 1
        code |= (1 if A[ty-1][tx-1] > center else 0) << 0

        # Since atomic add; adds value to the existing value
        # Need to figure out the fraction to be added in the previous value
        code = ( code - center)

        A[ty,tx] = code

        cuda.syncthreads()

        # Fun It's Fun to have a visible LBP Texture
        # So, overriding that with the origional vale.
        val  = A[ty,tx]
        cuda.atomic.add(arry, (x,y),val)
        cuda.syncthreads()

        # This Atomic Operation is equivalent to  hist[code % 256] += 1
        ind = code % BIN_COUNT
        cuda.atomic.add(hist, ind, 1)



def main():

    gray = misc.face(gray=True).astype(np.float32)
    # gray =rgb2gray(imread('test2.jpeg'))
    # gray =rgb2gray(image)
    src1 = resize(gray, (n,n), order=1, preserve_range=True).astype(np.int32);

    plt.imshow(src1, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    histogram = np.zeros(BIN_COUNT, dtype=np.int32)

    # We have threadCount*threadCount per block
    threadsperblock = (threadCount,threadCount)

    blockspergrid_x = math.ceil(src1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(src1.shape[1] / threadsperblock[1])
    # We have 2*2 blocks
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(threadsperblock)
    print(blockspergrid)

    cstream = cuda.stream()  # use stream to trigger async memory transfer
    ts = timer()
    # Increase Counter to measure the Efficiency
    count = 1
    for i in range(count):
        with cstream.auto_synchronize():
            # Copies Data to the device.
            d_src1 = cuda.to_device(src1, stream=cstream)
            # Copies histogram.
            d_hist_src = cuda.to_device(histogram, stream=cstream)
            # call the kernel fucntion
            lbp_texture[blockspergrid, threadsperblock, cstream](d_src1,d_hist_src)

            d_src1.copy_to_host(src1, stream=cstream)
            d_hist_src.copy_to_host(histogram, stream=cstream)

    te = timer()
    print('GPU Process ',count," Iterations : in ", te - ts)
    print('histogram is')
    print(histogram)

    plt.imshow(src1, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()


print('done')
