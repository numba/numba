import numpy as np
from numba import njit, types
from unittest import TestCase
from numba.tests.support import override_env_config

_DEBUG = False
if _DEBUG:
    from llvmlite import binding as llvm
    # Prints debug info from the LLVMs vectorizer
    llvm.set_option("", "--debug-only=loop-vectorize")


class TestVectorization(TestCase):
    """
    Tests to assert that code which should vectorize does indeed vectorize
    """
    def gen_ir(self, func, args_tuple, **flags):
        with override_env_config(
            "NUMBA_CPU_NAME", "skylake-avx512"
        ), override_env_config("NUMBA_CPU_FEATURES", ""):
            jobj = njit(**flags)(func)
            jobj.compile(args_tuple)
            ol = jobj.overloads[jobj.signatures[0]]
            return ol.library.get_llvm_str()

    def test_nditer_loop(self):
        # see https://github.com/numba/numba/issues/5033
        def do_sum(x):
            acc = 0
            for v in np.nditer(x):
                acc += v.item()
            return acc

        llvm_ir = self.gen_ir(do_sum, (types.float64[::1],), fastmath=True)
        self.assertIn("vector.body", llvm_ir)
        self.assertIn("llvm.loop.isvectorized", llvm_ir)

    def test_slp(self):
        # Sample translated from:
        # https://www.llvm.org/docs/Vectorizers.html#the-slp-vectorizer

        def foo(a1, a2, b1, b2, A):
            A[0] = a1 * (a1 + b1)
            A[1] = a2 * (a2 + b2)
            A[2] = a1 * (a1 + b1)
            A[3] = a2 * (a2 + b2)

        ty = types.float64
        llvm_ir = self.gen_ir(foo, ((ty,) * 4 + (ty[::1],)), fastmath=True)
        self.assertIn("2 x double", llvm_ir)
