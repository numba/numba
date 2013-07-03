import numpy as np
from .support import testcase, main
from numbapro import cuda, uint32

def atomic_add(ary):
    tid = cuda.threadIdx.x
    sm = cuda.shared.array(32, uint32)
    sm[tid] = 0
    cuda.syncthreads()
    bin = ary[tid] % 32
    cuda.atomic.add(sm, bin, 1)
    cuda.syncthreads()
    ary[tid] = sm[tid]

@testcase
def test_atomic_add():
    ary = np.random.randint(0, 32, size=32).astype(np.uint32)
    orig = ary.copy()
    cuda_atomic_add = cuda.jit('void(uint32[:])')(atomic_add)
    cuda_atomic_add[1, 32](ary)

    gold = np.zeros(32, dtype=np.uint32)
    for i in range(orig.size):
        gold[orig[i]] += 1

    assert np.all(ary == gold)


if __name__ == '__main__':
    main()
