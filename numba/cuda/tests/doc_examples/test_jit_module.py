# Contents in this file are referenced from the sphinx-generated docs.
# "magictoken" is used for markers as beginning and ending of example text.

import unittest
from numba.cuda.compiler import DeviceFunctionTemplate, Dispatcher
from numba.cuda.testing import CUDATestCase, skip_on_cudasim


@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestJitModule(CUDATestCase):
    def test_ex_multiple_jit_module(self):
        # magictoken.ex_multiple_jit_module.begin
        from numba import cuda
        import numpy as np

        # Bare functions that should be jitted as device functions

        def inc(x):
            return x + 1

        def add(x, y):
            return x + y

        # Compile all preceding undecorated functions as device functions
        cuda.jit_module()

        # Bare functions that should be jitted as kernels

        def call_device_inc(x):
            x[0] = inc(x[0])

        def call_device_add(x, y):
            x[0] = add(x[0], y[0])

        def inc_add(x):
            y = inc(x[0])
            x[0] = add(x[0], y)

        # Compile all preceding undecorated functions as kernel functions
        cuda.jit_module(device=False)

        # Host functions to call kernels. These do not get jitted because they follow
        # the calls to `jit_module`.

        def call_inc(x):
            x = np.array([x])
            call_device_inc[1, 1](x)
            return x[0]

        def call_add(x, y):
            x = np.array([x])
            y = np.array([y])
            call_device_add[1, 1](x, y)
            return x[0]

        def call_inc_add(x):
            x = np.array([x])
            inc_add[1, 1](x)
            return x[0]
        # magictoken.ex_multiple_jit_module.end

        # Hidden from the example as it is only needed for testing
        def call_py_inc_add(x):
            # We can't call inc_add directly for testing as a Python function,
            # because it calls inc and add - these are Dispatchers, not Python
            # functions. We manually construct the expected result using
            # inc.py_func and add.py_func instead.
            y = inc.py_func(x)
            return add.py_func(x, y)

            self.assertIsInstance(inc, DeviceFunctionTemplate)
            self.assertIsInstance(add, DeviceFunctionTemplate)
            self.assertIsInstance(inc_add, Dispatcher)

            # Test output of jitted functions is as expected.
            x, y = 1.7, 2.3
            self.assertEqual(call_inc(x),
                             inc.py_func(x))
            self.assertEqual(call_add(x, y),
                             add.py_func(x, y))
            self.assertEqual(call_inc_add(x),
                             call_py_inc_add(x))


if __name__ == '__main__':
    unittest.main()
