import numba.unittest_support as unittest

import numpy as np

from numba import config, jit, types
from numba.compiler import compile_isolated
from numba.tests.support import override_config, skip_py38_or_later


class TestJITMethod(unittest.TestCase):
    @skip_py38_or_later
    def test_bound_jit_method_with_loop_lift(self):
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
        [cres] = something.method.overloads.values()
        jitloop = cres.lifted[0]
        [loopcres] = jitloop.overloads.values()
        self.assertTrue(loopcres.fndesc.native)

    def test_unbound_jit_method(self):
        class Something(object):
            def __init__(self, x0):
                self.x0 = x0

            @jit
            def method(self):
                return self.x0

        something = Something(3)
        self.assertEquals(Something.method(something), 3)


class TestDisabledJIT(unittest.TestCase):
    def test_decorated_function(self):
        with override_config('DISABLE_JIT', True):
            def method(x):
                return x
            jitted = jit(method)

        self.assertEqual(jitted, method)
        self.assertEqual(10, method(10))
        self.assertEqual(10, jitted(10))

    def test_decorated_function_with_kwargs(self):
        with override_config('DISABLE_JIT', True):
            def method(x):
                return x
            jitted = jit(nopython=True)(method)

        self.assertEqual(jitted, method)
        self.assertEqual(10, method(10))
        self.assertEqual(10, jitted(10))

if __name__ == '__main__':
    unittest.main()
