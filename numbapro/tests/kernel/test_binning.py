import numpy as np
from numbapro.parallel.kernel import CU

def zero(tid, ary):
    ary[tid] = 0

def binning(tid, inA, inB, outC, outD):
    vala = inA[tid]
    valb = inB[tid]
    if vala > valb:
        outC[tid] = vala
    else:
        outD[tid] = valb

def test():
    n = 1000000
    np.random.seed(0xcafe)
    A = np.random.random(n)
    B = np.random.random(n)
    print A
    print B
    
    C = np.empty(n)
    D = np.empty(n)

    cu = CU('gpu')

    dA = cu.input(A)
    dB = cu.input(B)
    dC = cu.output(C)
    dD = cu.output(D)

    cu.enqueue(zero, ntid=C.size, args=(dC,))
    cu.enqueue(zero, ntid=D.size, args=(dD,))
    cu.enqueue(binning, ntid=C.size, args=(dA, dB, dC, dD))
    
    cu.wait()
    print C.size, C
    print D.size, D

    cu.close()

if __name__ == '__main__':
    test()
