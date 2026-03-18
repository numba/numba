from typing import List
from dataclasses import dataclass, field
from numba import cuda, float32
from numba.cuda.compiler import compile_ptx_for_current_device
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import unittest


@dataclass
class FMACriterion:
    fma_expected: List[str] = field(default_factory=list)
    fma_unexpected: List[str] = field(default_factory=list)
    nofma_expected: List[str] = field(default_factory=list)
    nofma_unexpected: List[str] = field(default_factory=list)

    def check(self, test: CUDATestCase, fma_ptx: str, nofma_ptx: str):
        test.assertTrue(all(i in fma_ptx for i in self.fma_expected))
        test.assertTrue(all(i not in fma_ptx for i in self.fma_unexpected))
        test.assertTrue(all(i in nofma_ptx for i in self.nofma_expected))
        test.assertTrue(all(i not in nofma_ptx for i in self.nofma_unexpected))


@skip_on_cudasim('FMA option and PTX inspection not available on cudasim')
class TestFMAOption(CUDATestCase):
    def _test_fma_common(self, pyfunc, sig, device, criterion):
        # Test jit code path
        fmaver = cuda.jit(sig, device=device)(pyfunc)
        nofmaver = cuda.jit(sig, device=device, fma=False)(pyfunc)

        criterion.check(
            self, fmaver.inspect_asm(sig), nofmaver.inspect_asm(sig)
        )

        # Test compile_ptx code path
        fmaptx, _ = compile_ptx_for_current_device(pyfunc, sig, device=device)
        nofmaptx, _ = compile_ptx_for_current_device(
            pyfunc, sig, device=device, fma=False
        )

        criterion.check(self, fmaptx, nofmaptx)

    def _test_fma_unary(self, op, criterion):
        def kernel(r, x):
            r[0] = op(x)

        def device_function(x):
            return op(x)

        self._test_fma_common(kernel, (float32[::1], float32), device=False,
                              criterion=criterion)
        self._test_fma_common(device_function, (float32,), device=True,
                              criterion=criterion)

    def test_muladd(self):
        # x * y + z is the canonical FMA candidate: the compiler should
        # contract it into a single fma.rn.f32 instruction by default.
        def kernel(r, x, y, z):
            r[0] = x * y + z

        def device_function(x, y, z):
            return x * y + z

        criterion = FMACriterion(
            fma_expected=['fma.rn.f32'],
            nofma_unexpected=['fma.rn.f32'],
        )

        self._test_fma_common(
            kernel,
            (float32[::1], float32, float32, float32),
            device=False,
            criterion=criterion,
        )
        self._test_fma_common(
            device_function,
            (float32, float32, float32),
            device=True,
            criterion=criterion,
        )


if __name__ == '__main__':
    unittest.main()
