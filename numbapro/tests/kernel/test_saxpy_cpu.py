import numpy as np
from numbapro.parallel.kernel import CU

# do saxpy in two kernels

def product(tid, A, B, Prod):
    Prod[tid] = A[tid] * B[tid]

def sum(tid, A, B, Sum):
    Sum[tid] = A[tid] + B[tid]

def test():
    n = 10000000
    A = np.arange(n)
    B = A #np.arange(n)
    C = A #np.arange(n)
    print A
    print B
    print C

    D = np.empty(n)

    cu = CU('cpu')

    dA = cu.input(A)
    dB = cu.input(B)
    dC = cu.input(C)
    dProd = cu.scratch_like(D)
    dSum  = cu.output(D)

    from timeit import default_timer as timer

    # warm up
    cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
    cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))
    cu.wait()

    # real deal
    ts = timer()
    cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
    cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))

    cu.wait()
    te = timer()
    tcu = te - ts

    print D.size, D

    ts = timer()
    exp = A * B + C
    te = timer()
    tnp = te - ts

    print 'time'
    print 'numpy:', tnp
    print 'cu   :', tcu

    assert np.allclose(A * B + C, D)

    cu.close()

if __name__ == '__main__':
    test()
