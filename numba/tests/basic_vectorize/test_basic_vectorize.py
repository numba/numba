import numpy as np
from numba import *
from numba.vectorize import Vectorize
import unittest
import logging

logger = logging.getLogger(__name__)

def vector_add(a, b):
    return a + b

class TestBasicVectorize(unittest.TestCase):
    def test_basic_vectorize_bytecode(self):
        # deprecated support; safe to remove in the future
        self._test_template('bytecode')

    def test_basic_vectorize_ast(self):
        self._test_template('ast')

    def _test_template(self, backend):
        # build basic native code ufunc
        bv = Vectorize(vector_add, backend=backend)
        bv.add(restype=int32, argtypes=[int32, int32])
        bv.add(restype=uint32, argtypes=[uint32, uint32])
        bv.add(restype=f4, argtypes=[f4, f4])
        bv.add(restype=f8, argtypes=[f8, f8])
        basic_ufunc = bv.build_ufunc()
        # build python ufunc
        np_ufunc = np.add
        # test it out
        def test(ty):
            logger.debug("backend = %s | dtype = %s", backend, ty)
            data = np.linspace(0., 10000., 100000).astype(ty)
            result = basic_ufunc(data, data)
            gold = np_ufunc(data, data)
            # check result
            np.allclose(gold, result)

        test(np.double)
        test(np.float32)
        test(np.int32)
        test(np.uint32)

if __name__ == '__main__':
    unittest.main()
