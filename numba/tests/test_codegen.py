"""
Tests for numba.targets.codegen.
"""

from __future__ import print_function

import base64
import ctypes
import pickle
import subprocess
import sys

import llvmlite.binding as ll

import numba.unittest_support as unittest
from numba import utils
from numba.targets.codegen import JITCPUCodegen
from .support import TestCase


asm_sum = r"""
    define i32 @sum(i32 %.1, i32 %.2) {
      %.3 = add i32 %.1, %.2
      ret i32 %.3
    }
    """

ctypes_sum_ty = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_int)


class JITCPUCodegenTestCase(TestCase):
    """
    Test the JIT code generation.
    """

    def setUp(self):
        self.codegen = JITCPUCodegen('test_codegen')

    def compile_module(self, asm):
        ll_module = ll.parse_assembly(asm)
        ll_module.verify()
        library = self.codegen.create_library('compiled_module')
        library.add_llvm_module(ll_module)
        return library

    @classmethod
    def _check_unserialize_sum(cls, state):
        codegen = JITCPUCodegen('other_codegen')
        library = codegen.unserialize_library(state)
        ptr = library.get_pointer_to_function("sum")
        cfunc = ctypes_sum_ty(ptr)
        res = cfunc(2, 3)
        assert res == 5, res

    def test_get_pointer_to_function(self):
        library = self.compile_module(asm_sum)
        ptr = library.get_pointer_to_function("sum")
        self.assertIsInstance(ptr, utils.integer_types)
        cfunc = ctypes_sum_ty(ptr)
        self.assertEqual(cfunc(2, 3), 5)

    def test_serialize_unserialize(self):
        library = self.compile_module(asm_sum)
        state = library.serialize()
        self._check_unserialize_sum(state)

    def test_unserialize_other_process(self):
        library = self.compile_module(asm_sum)
        state = library.serialize()
        arg = base64.b64encode(pickle.dumps(state, -1))
        code = """if 1:
            import base64
            import pickle
            import sys
            from numba.tests.test_codegen import %(test_class)s

            state = pickle.loads(base64.b64decode(sys.argv[1]))
            %(test_class)s._check_unserialize_sum(state)
            """ % dict(test_class=self.__class__.__name__)
        subprocess.check_call([sys.executable, '-c', code, arg.decode()])

    def test_magic_tuple(self):
        tup = self.codegen.magic_tuple()
        pickle.dumps(tup)
        cg2 = JITCPUCodegen('xxx')
        self.assertEqual(cg2.magic_tuple(), tup)


if __name__ == '__main__':
    unittest.main()
