from __future__ import print_function

import contextlib
import imp
import os
import sys
from ctypes import *

from numba import unittest_support as unittest
from numba.pycc import find_shared_ending, find_pyext_ending, main
from numba.pycc.decorators import clear_export_registry
from .support import TestCase

from numba.tests.support import static_temp_directory

base_path = os.path.dirname(os.path.abspath(__file__))


def unset_macosx_deployment_target():
    """Unset MACOSX_DEPLOYMENT_TARGET because we are not building portable
    libraries
    """
    macosx_target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
    if macosx_target is not None:
        del os.environ['MACOSX_DEPLOYMENT_TARGET']


class BasePYCCTest(TestCase):

    def setUp(self):
        self.tmpdir = static_temp_directory('test_pycc')

    def tearDown(self):
        # Since we're executing the module-under-test several times
        # from the same process, we must clear the exports registry
        # between invocations.
        clear_export_registry()

    @contextlib.contextmanager
    def check_c_ext(self, extdir, name):
        sys.path.append(extdir)
        try:
            lib = __import__(name)
            yield lib
        finally:
            sys.path.remove(extdir)
            sys.modules.pop(name, None)


class TestLegacyAPI(BasePYCCTest):

    def test_pycc_ctypes_lib(self):
        """
        Test creating a C shared library object using pycc.
        """
        unset_macosx_deployment_target()

        source = os.path.join(base_path, 'compile_with_pycc.py')
        cdll_modulename = 'test_dll_legacy' + find_shared_ending()
        cdll_path = os.path.join(self.tmpdir, cdll_modulename)
        if os.path.exists(cdll_path):
            os.unlink(cdll_path)

        main(args=['--debug', '-o', cdll_path, source])
        lib = CDLL(cdll_path)
        lib.mult.argtypes = [POINTER(c_double), c_void_p, c_void_p,
                             c_double, c_double]
        lib.mult.restype = c_int

        lib.multf.argtypes = [POINTER(c_float), c_void_p, c_void_p,
                              c_float, c_float]
        lib.multf.restype = c_int

        res = c_double()
        lib.mult(byref(res), None, None, 123, 321)
        self.assertEqual(res.value, 123 * 321)

        res = c_float()
        lib.multf(byref(res), None, None, 987, 321)
        self.assertEqual(res.value, 987 * 321)

    def test_pycc_pymodule(self):
        """
        Test creating a CPython extension module using pycc.
        """
        unset_macosx_deployment_target()

        source = os.path.join(base_path, 'compile_with_pycc.py')
        modulename = 'test_pyext_legacy'
        out_modulename = os.path.join(self.tmpdir,
                                      modulename + find_pyext_ending())
        if os.path.exists(out_modulename):
            os.unlink(out_modulename)

        main(args=['--debug', '--python', '-o', out_modulename, source])

        with self.check_c_ext(self.tmpdir, modulename) as lib:
            res = lib.multi(123, 321)
            self.assertPreciseEqual(res, 123 * 321)
            res = lib.multf(987, 321)
            self.assertPreciseEqual(res, 987.0 * 321.0)

    def test_pycc_bitcode(self):
        """
        Test creating a LLVM bitcode file using pycc.
        """
        unset_macosx_deployment_target()

        modulename = os.path.join(base_path, 'compile_with_pycc')
        bitcode_modulename = os.path.join(self.tmpdir, 'test_bitcode_legacy.bc')
        if os.path.exists(bitcode_modulename):
            os.unlink(bitcode_modulename)

        main(args=['--debug', '--llvm', '-o', bitcode_modulename,
                   modulename + '.py'])

        # Sanity check bitcode file contents
        with open(bitcode_modulename, "rb") as f:
            bc = f.read()

        bitcode_wrapper_magic = b'\xde\xc0\x17\x0b'
        bitcode_magic = b'BC\xc0\xde'
        self.assertTrue(bc.startswith((bitcode_magic, bitcode_wrapper_magic)), bc)


class TestCC(BasePYCCTest):

    def setUp(self):
        super(TestCC, self).setUp()
        from . import compile_with_pycc
        self._test_module = compile_with_pycc
        imp.reload(self._test_module)

    def test_cc_properties(self):
        cc = self._test_module.cc
        self.assertEqual(cc.name, 'pycc_test_output')

        # Inferred output directory
        d = self._test_module.cc.output_dir
        self.assertTrue(os.path.isdir(d), d)

        # Inferred output filename
        f = self._test_module.cc.output_file
        self.assertFalse(os.path.exists(f), f)
        self.assertTrue(os.path.basename(f).startswith('pycc_test_output.'), f)
        if sys.platform == 'linux':
            self.assertTrue(f.endswith('.so'), f)
            if sys.version_info >= (3,):
                self.assertIn('.cpython', f)

    def test_compile(self):
        cc = self._test_module.cc
        cc.debug = True
        cc.output_dir = self.tmpdir
        cc.compile()

        with self.check_c_ext(self.tmpdir, cc.name) as lib:
            res = lib.multi(123, 321)
            self.assertPreciseEqual(res, 123 * 321)
            res = lib.multf(987, 321)
            self.assertPreciseEqual(res, 987.0 * 321.0)
            res = lib.square(5)
            self.assertPreciseEqual(res, 25)


if __name__ == "__main__":
    unittest.main()
