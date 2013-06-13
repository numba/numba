from .support import testcase, main, run
from numbapro import cuda
from numbapro.cudapipeline import cudapy
from numbapro.cudapipeline.npm.types import *

def threadidx(ary):
    i = cuda.threadIdx.x
    ary[0] = i

@testcase
def test_threadidx():
    compiled = cudapy.compile_kernel(threadidx, [arraytype(int32, 1, 'C')])

if __name__ == '__main__':
    main()
