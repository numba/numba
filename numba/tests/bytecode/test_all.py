#! /usr/bin/env python
# ______________________________________________________________________
'''test_all

Grand unified unit test script for Numba.
'''
# ______________________________________________________________________

import unittest

from .test_avg2d import TestAvg2D
from .test_cfg import TestCFG
#from test_complex import TestComplex
#from test_extern_call import TestExternCall
from .test_filter2d import TestFilter2d
from .test_forloop import TestForLoop
from .test_if import TestIf
from .test_indexing import TestIndexing
from .test_issues import TestIssues
#from test_mandelbrot import TestMandelbrot
from .test_multiarray_api import TestMultiarrayAPI
#from test_tuple import TestTuple # Not properly refcounted in bytecode translator!
#from test_vectorize import TestVectorize
from .test_while import TestWhile
from .test_sum import TestSum2d
#from test_print import TestPrint

# ______________________________________________________________________

if __name__ == "__main__":
    unittest.main()

# ______________________________________________________________________
# End of test_all.py
