from numba.cuda.testing import skip_on_cudasim
from numba import cuda
from numba.core import types
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

        self._check(foo, sig=(types.int32[:],), expect=False)

    def test_lineinfo_in_asm(self):
        @cuda.jit(lineinfo=True)
        def foo(x):
            x[0] = 1

        self._check(foo, sig=(types.int32[:],), expect=True)


if __name__ == '__main__':
    unittest.main()
