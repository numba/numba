"""
Tests for numba.targets.codegen.
"""

from __future__ import print_function

import warnings
import base64
import ctypes
import pickle
import subprocess
import sys
import weakref

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

# Note we're using a rather mangled function name to check that it
# is compatible with object serialization.

asm_sum_inner = """
    define i32 @"__main__.ising_element_update$1.array(int8,_2d,_C).int64.int64"(i32 %.1, i32 %.2) {
      %.3 = add i32 %.1, %.2
      ret i32 %.3
    }
"""

asm_sum_outer = """
    declare i32 @"__main__.ising_element_update$1.array(int8,_2d,_C).int64.int64"(i32 %.1, i32 %.2)

    define i32 @sum(i32 %.1, i32 %.2) {
      %.3 = call i32 @"__main__.ising_element_update$1.array(int8,_2d,_C).int64.int64"(i32 %.1, i32 %.2)
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

    def tearDown(self):
        del self.codegen

    def compile_module(self, asm, linking_asm=None):
        library = self.codegen.create_library('compiled_module')
        ll_module = ll.parse_assembly(asm)
        ll_module.verify()
        library.add_llvm_module(ll_module)
        if linking_asm:
            linking_library = self.codegen.create_library('linking_module')
            ll_module = ll.parse_assembly(linking_asm)
            ll_module.verify()
            linking_library.add_llvm_module(ll_module)
            library.add_linking_library(linking_library)
        return library

    @classmethod
    def _check_unserialize_sum(cls, state):
        codegen = JITCPUCodegen('other_codegen')
        library = codegen.unserialize_library(state)
        ptr = library.get_pointer_to_function("sum")
        assert ptr, ptr
        cfunc = ctypes_sum_ty(ptr)
        res = cfunc(2, 3)
        assert res == 5, res

    def test_get_pointer_to_function(self):
        library = self.compile_module(asm_sum)
        ptr = library.get_pointer_to_function("sum")
        self.assertIsInstance(ptr, utils.integer_types)
        cfunc = ctypes_sum_ty(ptr)
        self.assertEqual(cfunc(2, 3), 5)
        # Note: With llvm3.9.1, deleting `library` will cause memory error in
        #       the following code during running of optimization passes in
        #       LLVM. The reason of the error is unclear. The error is known to
        #       replicate on osx64 and linux64.

        # Same, but with dependency on another library
        library2 = self.compile_module(asm_sum_outer, asm_sum_inner)
        ptr = library2.get_pointer_to_function("sum")
        self.assertIsInstance(ptr, utils.integer_types)
        cfunc = ctypes_sum_ty(ptr)
        self.assertEqual(cfunc(2, 3), 5)

    def test_magic_tuple(self):
        tup = self.codegen.magic_tuple()
        pickle.dumps(tup)
        cg2 = JITCPUCodegen('xxx')
        self.assertEqual(cg2.magic_tuple(), tup)

    # Serialization tests.

    def _check_serialize_unserialize(self, state):
        self._check_unserialize_sum(state)

    def _check_unserialize_other_process(self, state):
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

    def test_serialize_unserialize_bitcode(self):
        library = self.compile_module(asm_sum_outer, asm_sum_inner)
        state = library.serialize_using_bitcode()
        self._check_serialize_unserialize(state)

    def test_unserialize_other_process_bitcode(self):
        library = self.compile_module(asm_sum_outer, asm_sum_inner)
        state = library.serialize_using_bitcode()
        self._check_unserialize_other_process(state)

    def test_serialize_unserialize_object_code(self):
        library = self.compile_module(asm_sum_outer, asm_sum_inner)
        library.enable_object_caching()
        state = library.serialize_using_object_code()
        self._check_serialize_unserialize(state)

    def test_unserialize_other_process_object_code(self):
        library = self.compile_module(asm_sum_outer, asm_sum_inner)
        library.enable_object_caching()
        state = library.serialize_using_object_code()
        self._check_unserialize_other_process(state)

    def test_cache_disabled_inspection(self):
        """
        """
        library = self.compile_module(asm_sum_outer, asm_sum_inner)
        library.enable_object_caching()
        state = library.serialize_using_object_code()

        # exercise the valid behavior
        with warnings.catch_warnings(record=True) as w:
            old_llvm = library.get_llvm_str()
            old_asm = library.get_asm_str()
            library.get_function_cfg('sum')
        self.assertEqual(len(w), 0)

        # unserialize
        codegen = JITCPUCodegen('other_codegen')
        library = codegen.unserialize_library(state)

        # the inspection methods would warn and give incorrect result
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertNotEqual(old_llvm, library.get_llvm_str())
        self.assertEqual(len(w), 1)
        self.assertIn("Inspection disabled", str(w[0].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.assertNotEqual(library.get_asm_str(), old_asm)
        self.assertEqual(len(w), 1)
        self.assertIn("Inspection disabled", str(w[0].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with self.assertRaises(NameError) as raises:
                library.get_function_cfg('sum')
        self.assertEqual(len(w), 1)
        self.assertIn("Inspection disabled", str(w[0].message))
        self.assertIn("sum", str(raises.exception))

    # Lifetime tests

    @unittest.expectedFailure  # MCJIT removeModule leaks and it is disabled
    def test_library_lifetime(self):
        library = self.compile_module(asm_sum_outer, asm_sum_inner)
        # Exercise code generation
        library.enable_object_caching()
        library.serialize_using_bitcode()
        library.serialize_using_object_code()
        u = weakref.ref(library)
        v = weakref.ref(library._final_module)
        del library
        # Both the library and its backing LLVM module are collected
        self.assertIs(u(), None)
        self.assertIs(v(), None)


if __name__ == '__main__':
    unittest.main()
