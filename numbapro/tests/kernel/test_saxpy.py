import numpy as np
from numbapro.parallel.kernel import CU

# do saxpy in two kernels

def product(tid, A, B, Prod):
    Prod[tid] = A[tid] * B[tid]

def sum(tid, A, B, Sum):
    Sum[tid] = A[tid] + B[tid]

def test():
    n = 100
    A = np.arange(n)
    B = np.arange(n)
    C = np.arange(n)
    print A
    print B
    print C

    D = np.empty(n)

    cu = CU('gpu')

    dA = cu.input(A)
    dB = cu.input(B)
    dC = cu.input(C)
    dProd = cu.scratch_like(D)
    dSum  = cu.output(D)

    cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
    cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))

    cu.wait()
    print D.size, D

    cu.close()

if __name__ == '__main__':
    test()
