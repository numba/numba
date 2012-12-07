'''
Test parallel-vectorize with numpy.fromfunc.
Uses the work load from test_parallel_vectorize.
'''

from numbapro.vectorize.parallel import *
#from numba.vectorize._internal import fromfunc
from numbapro.vectorize._numba_vectorize._internal import fromfunc
import numpy as np
import unittest
from test_parallel_vectorize import Work_D_D

class TestInner_Numpy(unittest.TestCase):
    def test_numpy_1(self):
        module = Module.new(__name__)

        spufdef = SpecializedParallelUFunc(ParallelUFuncPlatform(num_thread=2),
                                           UFuncCoreGeneric(Work_D_D()(module)))

        sppufunc = spufdef(module)

        module.verify()

        mpm = PassManager.new()
        pmbuilder = PassManagerBuilder.new()
        pmbuilder.opt_level = 3
        pmbuilder.populate(mpm)

        mpm.run(module)
        #    print module

        # run

        exe = CExecutor(module)
        funcptr = exe.engine.get_pointer_to_function(sppufunc)
        #   print("Function pointer: %x" % funcptr)

        ptr_t = long # py2 only

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        typenum = np.dtype(np.double).num

        ufunc = fromfunc([ptr_t(funcptr)], [[typenum, typenum]], 1, 1, [None])

        x = np.linspace(0., 10., 1000).astype(np.double)
        #    print x
        ans = ufunc(x)
        #    print ans

        self.assertTrue(( ans == x/2.345 ).all())

if __name__ == '__main__':
    unittest.main()
