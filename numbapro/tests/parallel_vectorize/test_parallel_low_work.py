'''
There was a deadlock problem when work count is smaller than number of threads.
'''

import numpy as np
from numba import *
from numbapro.vectorize.parallel import ParallelVectorize
from time import time
import unittest

def vector_add(a, b):
    return a + b

class TestParallelLowWorkCount(unittest.TestCase):
    def test_low_workcount(self):
        # build parallel native code ufunc
        pv = ParallelVectorize(vector_add)
        pv.add(restype=int32, argtypes=[int32, int32])
        pv.add(restype=uint32, argtypes=[uint32, uint32])
        pv.add(restype=f, argtypes=[f, f])
        pv.add(restype=d, argtypes=[d, d])
        para_ufunc = pv.build_ufunc()

        # build python ufunc
        np_ufunc = np.vectorize(vector_add)

        # test it out
        def test(ty):
            #print("Test %s" % ty)
            data = np.arange(1).astype(ty) # just one item

            ts = time()
            result = para_ufunc(data, data)
            tnumba = time() - ts

            ts = time()
            gold = np_ufunc(data, data)
            tnumpy = time() - ts

            #print("Numpy time: %fs" % tnumpy)
            #print("Numba time: %fs" % tnumba)

            #if tnumba < tnumpy:
            #    print("Numba is FASTER by %fx" % (tnumpy/tnumba))
            #else:
            #    print("Numba is SLOWER by %fx" % (tnumba/tnumpy))


            for expect, got in zip(gold, result):
                self.assertLess(abs(got - expect) / (expect + 1e-31), 1e-6)

        test(np.double)
        test(np.float32)
        test(np.int32)
        test(np.uint32)

if __name__ == '__main__':
    unittest.main()
