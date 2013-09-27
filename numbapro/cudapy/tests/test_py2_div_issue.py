from .support import testcase, main
from numbapro import cuda, jit, float32, int32
import numpy as np

def test_py2_div_issue():
    @jit(argtypes=[float32[:], float32[:], float32[:], int32], target='gpu')
    def preCalc(y, yA, yB, numDataPoints, z):
        i = cuda.grid(1)
        k = i % numDataPoints

        ans = float32(1.001 * float32(i))

        y[i] = ans
        yA[i] = ans * 1.0
        yB[i] = ans / 1.0


    numDataPoints = 15

    y = np.zeros(numDataPoints, dtype=np.float32)
    yA = np.zeros(numDataPoints, dtype=np.float32)
    yB = np.zeros(numDataPoints, dtype=np.float32)
    z = 1.0
    preCalc[1, 15](y, yA, yB, numDataPoints)

    print 'y'
    print y

    print 'yA'
    print yA

    print 'yB'
    print yB

    assert(np.all(y == yA == yB))

