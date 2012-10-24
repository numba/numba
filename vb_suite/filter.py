import time

from scipy.misc import lena
from vbench.benchmark import Benchmark

common_setup = """
from numba import *
"""

setup = common_setup + """
@jit(restype=void, argtypes=[double[:,:], double[:,:], double[:,:]],
     backend='ast')
def filter(image, filt, output):
    M, N = image.shape
    m, n = filt.shape
    for i in range(m/2, M-m/2):
        for j in range(n/2, N-n/2):
            result = 0.0
            for k in range(m):
                for l in range(n):
                    result += image[i+k-m/2,j+l-n/2]*filt[k, l]
            output[i,j] = result

image = lena().astype('double')
filt = np.ones((15,15),dtype='double')
filt /= filt.sum()
output = image.copy()
"""
stmt = "filter(image, filt, output)"
bench = Benchmark(stmt, setup, name="filter")