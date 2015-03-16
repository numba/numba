from __future__ import print_function, absolute_import
import numpy as np
from numba import cuda, float32, int32
from numba.cuda.testing import unittest


class TestCudaPy2Div(unittest.TestCase):
    def test_py2_div_issue(self):
        @cuda.jit(argtypes=[float32[:], float32[:], float32[:], int32])
        def preCalc(y, yA, yB, numDataPoints):
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

        self.assertTrue(np.all(y == yA))
        self.assertTrue(np.all(y == yB))


if __name__ == '__main__':
    unittest.main()
