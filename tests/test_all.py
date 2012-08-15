#! /usr/bin/env python
# ______________________________________________________________________
'''test_all

Grand unified unit test script for Numba.
'''
# ______________________________________________________________________

import unittest

from test_cfg import TestCFG
from test_complex import TestComplex
from test_extern_call import TestExternCall
from test_filter2d import TestFilter2d
from test_forloop import TestForLoop
from test_getattr import TestGetattr
from test_if import TestIf
from test_indexing import TestIndexing
from test_mandelbrot import TestMandelbrot
from test_multiarray_api import TestMultiarrayAPI
from test_tuple import TestTuple
#from test_vectorize import TestVectorize
from test_while import TestWhile
from test_sum import TestSum2d

# ______________________________________________________________________

if __name__ == "__main__":
    print type(__builtins__)
    unittest.main()

# ______________________________________________________________________
# End of test_all.py
