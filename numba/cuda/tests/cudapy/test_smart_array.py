from __future__ import print_function, division, absolute_import

import sys

import numpy as np

from numba import unittest_support as unittest
from numba import types
from numba.extending import typeof_impl
from numba.cuda.kernels.transpose import transpose
from numba.tracing import event
from numba import SmartArray
from numba.cuda.testing import skip_on_cudasim

@skip_on_cudasim('Simulator does not support Device arrays')
class TestJIT(unittest.TestCase):
    """Test handling of numba.SmartArray"""

    def test_transpose(self):
        
        # To verify non-redundant data movement run this test with NUMBA_TRACE=1
        a = SmartArray(np.arange(16, dtype=float).reshape(4,4))
        b = SmartArray(where='gpu', shape=(4,4), dtype=float)
        c = SmartArray(where='gpu', shape=(4,4), dtype=float)
        event("initialization done")
        transpose(a, b)
        event("checkpoint")
        transpose(b, c)
        event("done")
        self.assertTrue((c.host() == a.host()).all())

if __name__ == '__main__':
    unittest.main()
