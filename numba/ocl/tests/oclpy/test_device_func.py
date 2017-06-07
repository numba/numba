from __future__ import print_function, absolute_import, division


import types

import numpy as np

from numba.ocl.testing import unittest
from numba import ocl, jit
from numba.errors import TypingError


class TestDeviceFunc(unittest.TestCase):

    def test_use_add2f(self):

        @ocl.jit("float32(float32, float32)", device=True)
        def add2f(a, b):
            return a + b

        def use_add2f(ary):
            i = ocl.grid(1)
            ary[i] = add2f(ary[i], ary[i])

        compiled = ocl.jit("void(float32[:])")(use_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def test_indirect_add2f(self):

        @ocl.jit("float32(float32, float32)", device=True)
        def add2f(a, b):
            return a + b

        @ocl.jit("float32(float32, float32)", device=True)
        def indirect(a, b):
            return add2f(a, b)

        def indirect_add2f(ary):
            i = ocl.grid(1)
            ary[i] = indirect(ary[i], ary[i])

        compiled = ocl.jit("void(float32[:])")(indirect_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def _check_cpu_dispatcher(self, add):
        @ocl.jit
        def add_kernel(ary):
            i = ocl.grid(1)
            ary[i] = add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    def test_cpu_dispatcher(self):
        # Test correct usage
        @jit
        def add(a, b):
            return a + b

        self._check_cpu_dispatcher(add)

    def test_cpu_dispatcher_invalid(self):
        # Test invalid usage
        # Explicit signature disables compilation, which also disable
        # compiling on OCL.
        @jit('(i4, i4)')
        def add(a, b):
            return a + b

        # Check that the right error message is provided.
        with self.assertRaises(TypingError) as raises:
            self._check_cpu_dispatcher(add)
        msg = "Untyped global name 'add': using cpu function on device"
        self.assertIn(msg, str(raises.exception))

    def test_cpu_dispatcher_other_module(self):
        @jit
        def add(a, b):
            return a + b

        mymod = types.ModuleType(name='mymod')
        mymod.add = add
        del add

        @ocl.jit
        def add_kernel(ary):
            i = ocl.grid(1)
            ary[i] = mymod.add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)


if __name__ == '__main__':
    unittest.main()
