from numba import unittest_support as unittest
import numpy as np

try:
    from xnd import xnd
except ImportError:
    pass

class TestCase(unittest.TestCase):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.addTypeEqualityFunc(xnd, self.assertXndEqual)
        self.addTypeEqualityFunc(np.ndarray, self.assertNDArrayEqual)

    def assertXndEqual(self, x1, x2, msg=None):
        self.assertEqual(x1.type, x2.type, msg)
        np.testing.assert_allclose(x1, x2, rtol=1e-5, atol=1e-8, err_msg=msg)

    def assertNDArrayEqual(self, x1, x2, msg=None):
        np.testing.assert_allclose(x1, x2, rtol=1e-5, atol=1e-8, err_msg=msg)
