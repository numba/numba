from __future__ import print_function, absolute_import, division


import re
import types

import numpy as np

from numba.cuda.testing import unittest, skip_on_cudasim, SerialMixin
from numba import cuda, jit, int32
from numba.errors import TypingError


class TestDeviceFunc(SerialMixin, unittest.TestCase):

    def test_use_add2f(self):

        @cuda.jit("float32(float32, float32)", device=True)
        def add2f(a, b):
            return a + b

        def use_add2f(ary):
            i = cuda.grid(1)
            ary[i] = add2f(ary[i], ary[i])

        compiled = cuda.jit("void(float32[:])")(use_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def test_indirect_add2f(self):

        @cuda.jit("float32(float32, float32)", device=True)
        def add2f(a, b):
            return a + b

        @cuda.jit("float32(float32, float32)", device=True)
        def indirect(a, b):
            return add2f(a, b)

        def indirect_add2f(ary):
            i = cuda.grid(1)
            ary[i] = indirect(ary[i], ary[i])

        compiled = cuda.jit("void(float32[:])")(indirect_add2f)

        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)

        self.assertTrue(np.all(ary == exp), (ary, exp))

    def _check_cpu_dispatcher(self, add):
        @cuda.jit
        def add_kernel(ary):
            i = cuda.grid(1)
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

    @skip_on_cudasim('not supported in cudasim')
    def test_cpu_dispatcher_invalid(self):
        # Test invalid usage
        # Explicit signature disables compilation, which also disable
        # compiling on CUDA.
        @jit('(i4, i4)')
        def add(a, b):
            return a + b

        # Check that the right error message is provided.
        with self.assertRaises(TypingError) as raises:
            self._check_cpu_dispatcher(add)
        msg = "Untyped global name 'add':.*using cpu function on device"
        expected = re.compile(msg)
        self.assertTrue(expected.search(str(raises.exception)) is not None)

    def test_cpu_dispatcher_other_module(self):
        @jit
        def add(a, b):
            return a + b

        mymod = types.ModuleType(name='mymod')
        mymod.add = add
        del add

        @cuda.jit
        def add_kernel(ary):
            i = cuda.grid(1)
            ary[i] = mymod.add(ary[i], 1)

        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_ptx(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (int32, int32)
        cres = foo.compile(args)

        fname = cres.fndesc.mangled_name
        # Verify that the function name has "foo" in it as in the python name
        self.assertIn('foo', fname)

        ptx = foo.inspect_ptx(args)
        # Check that the compiled function name is in the PTX.
        self.assertIn(fname, ptx.decode('ascii'))

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_llvm(self):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (int32, int32)
        cres = foo.compile(args)

        fname = cres.fndesc.mangled_name
        # Verify that the function name has "foo" in it as in the python name
        self.assertIn('foo', fname)

        llvm = foo.inspect_llvm(args)
        # Check that the compiled function name is in the LLVM.
        self.assertIn(fname, llvm)


if __name__ == '__main__':
    unittest.main()
