from numba import unittest_support as unittest
import numpy as np

try:
    from xnd import xnd
except ImportError:
    xnd = None


@unittest.skipUnless(xnd, "requires xnd")
class TestCase(unittest.TestCase):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.addTypeEqualityFunc(xnd, self.assertXndEqual)

    def assertXndEqual(self, x1, x2, msg=None):
        self.assertEqual(x1.type, x2.type, msg)
        np.testing.assert_allclose(x1, x2, err_msg=msg)
