from numba.cuda.testing import skip_on_cudasim
from numba import cuda, float32, int32
from numba.cuda.testing import CUDATestCase
import re
import unittest


@skip_on_cudasim('Simulator does not produce lineinfo')
class TestCudaLineInfo(CUDATestCase):
    """
    These tests only check the compiled PTX for line mappings
    """
    def _getasm(self, fn, sig):
        fn.compile(sig)
        return fn.inspect_asm(sig)

    def _check(self, fn, sig, expect):
        asm = self._getasm(fn, sig=sig)
        # The name of this file should be present in the line mapping
        # if lineinfo was propagated through correctly.
        re_section_lineinfo = re.compile(r"test_lineinfo.py")
        match = re_section_lineinfo.search(asm)
        assertfn = self.assertIsNotNone if expect else self.assertIsNone
        assertfn(match, msg=asm)

    def test_no_lineinfo_in_asm(self):
        @cuda.jit(lineinfo=False)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(int32[:],), expect=False)

    def test_lineinfo_in_asm(self):
        @cuda.jit(lineinfo=True)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(int32[:],), expect=True)

    def test_lineinfo_maintains_error_model(self):
        sig = (float32[::1], float32[::1])

        @cuda.jit(sig, lineinfo=True)
        def divide_kernel(x, y):
            x[0] /= y[0]

        llvm = divide_kernel.inspect_llvm(sig)

        # When the error model is Python, the device function returns 1 to
        # signal an exception (e.g. divide by zero) has occurred. When the
        # error model is the default NumPy one (as it should be when only
        # lineinfo is enabled) the device function always returns 0.
        self.assertNotIn('ret i32 1', llvm)


if __name__ == '__main__':
    unittest.main()
