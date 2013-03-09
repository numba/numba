from __future__ import print_function
import sys
from contextlib import closing
import numpy as np
from numbapro.parallel.kernel import CU

# do saxpy in two kernels

def product(tid, A, B, Prod):
    Prod[tid] = A[tid] * B[tid]

def sum(tid, A, B, Sum):
    Sum[tid] = A[tid] + B[tid]

def saxpy(target='cpu'):
    if len(sys.argv) > 1:
        target = str(sys.argv[1])
    print('Using %s target' % target)
    n = 10000000
    A = np.arange(n)
    B = A.copy()
    C = A.copy()
    totaldatasize = A.size * 4.
    print('Total data size %.2fMB' % (totaldatasize / 2**20) )
#    print(A)
#    print(B)
#    print(C)

    D = np.empty(n)

    with closing(CU(target)) as cu:
        from timeit import default_timer as timer

        # warm up
        
        dA = cu.input(A)
        dB = cu.input(B)
        dC = cu.input(C)
        dProd = cu.scratch_like(D)
        dSum  = cu.output(D)

        cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
        cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))
        cu.wait()

        del dA, dB, dC, dProd, dSum
        # real deal

        ts = timer()

        dA = cu.input(A)
        dB = cu.input(B)
        dC = cu.input(C)
        dProd = cu.scratch_like(D)
        dSum  = cu.output(D)

        cu.enqueue(product, ntid=dProd.size, args=(dA, dB, dProd))
        cu.enqueue(sum, 	ntid=dSum.size,  args=(dProd, dC, dSum))

        cu.wait()
        te = timer()
        tcu = te - ts

#        print(D.size, D)

    # warm up
    exp = A * B + C

    ts = timer()
    exp = A * B + C
    te = timer()
    tnp = te - ts

    print('Verify')
    assert np.allclose(A * B + C, D)

    print('time'.center(50, '-'))
    print('numpy:', tnp)
    print('cu   :', tcu)

    print('throughput'.center(50, '-'))
    print('numpy:', totaldatasize / tnp / 2**20, 'MB/s')
    print('cu   :', totaldatasize / tcu / 2**20, 'MB/s')
    

if __name__ == '__main__':
    saxpy()
