import numba.unittest_support as unittest
from numba import jit
import numpy as np


class TestJITMethod(unittest.TestCase):
    def test_jit_method_with_loop_lift(self):

        class Something(object):
            def __init__(self, x0):
                self.x0 = x0

            @jit
            def method(self, x):
                a = np.empty(shape=5, dtype=np.float32)
                x0 = self.x0

                for i in range(a.shape[0]):
                    a[i] = x0 * x

                return a

        something = Something(3)
        np.testing.assert_array_equal(something.method(5),
            np.array([15, 15, 15, 15, 15], dtype=np.float32))

        # Check that loop lifting in nopython mode was successful
        [cres] = Something.method._compileinfos.values()
        jitloop = cres.lifted[0]
        [loopcres] = jitloop._compileinfos.values()
        self.assertTrue(loopcres.fndesc.native)

if __name__ == '__main__':
    unittest.main()
