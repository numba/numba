import numpy as np
from numbapro.parallel.kernel import CU

def incr(tid, A):
    A[tid] += 1

def test():
    A = np.arange(10)    
    A0 = A.copy()

    cu = CU('gpu')    
    dA = cu.inout(A)
    cu.enqueue(incr, ntid=A.size, args=(A,))

    print ' A', A
    print 'A0', A0 
    assert all(A == A0 + 1)

    cu.close()

if __name__ == '__main__':
    test()

