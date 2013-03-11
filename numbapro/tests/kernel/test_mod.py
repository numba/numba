import unittest
from contextlib import closing
import numpy as np
from numbapro import CU

def mod(tid, A, b):
    A[tid] = A[tid] % b

def test():
    n = 10
    A = np.arange(n)
    A0 = A.copy()
    cu = CU('cpu')
    with closing(cu):
        dA = cu.inout(A)
        cu.enqueue(mod, ntid=A.size, args=(A, n))
        cu.wait()
    assert np.allclose(A, A0 % n)

if __name__ == '__main__':
    test()
