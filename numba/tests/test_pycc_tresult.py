from __future__ import print_function

import os
import sys
from ctypes import *

from numba import unittest_support as unittest
from numba.pycc import find_shared_ending, find_pyext_ending, main
from numba.tests.support import static_temp_directory

base_path = os.path.dirname(os.path.abspath(__file__))


def unset_macosx_deployment_target():
    """Unset MACOSX_DEPLOYMENT_TARGET because we are not building portable
    libraries
    """
    macosx_target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
    if macosx_target is not None:
        del os.environ['MACOSX_DEPLOYMENT_TARGET']


class TestPYCC(unittest.TestCase):

    def setUp(self):
        self.tmpdir = static_temp_directory('test_pycc')

    def test_pycc_ctypes_lib(self):
        """
        Test creating a C shared library object using pycc.
        """
        unset_macosx_deployment_target()

        source = os.path.join(base_path, 'compile_with_pycc.py')
        cdll_modulename = 'test_dll' + find_shared_ending()
        cdll_path = os.path.join(self.tmpdir, cdll_modulename)
        if os.path.exists(cdll_path):
            os.unlink(cdll_path)

        main(args=['-o', cdll_path, source])
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
        modulename = 'pycc_test_pyext'
        out_modulename = os.path.join(self.tmpdir,
                                      modulename + find_pyext_ending())
        if os.path.exists(out_modulename):
            os.unlink(out_modulename)

        main(args=['--python', '-o', out_modulename, source])

        sys.path.append(self.tmpdir)
        try:
            lib = __import__(modulename)
        finally:
            sys.path.remove(self.tmpdir)
        try:
            res = lib.mult(123, 321)
            assert res == 123 * 321

            res = lib.multf(987, 321)
            assert res == 987 * 321
        finally:
            del lib

    def test_pycc_bitcode(self):
        """
        Test creating a LLVM bitcode file using pycc.
        """
        unset_macosx_deployment_target()

        modulename = os.path.join(base_path, 'compile_with_pycc')
        bitcode_modulename = os.path.join(self.tmpdir, 'test_bitcode.bc')
        if os.path.exists(bitcode_modulename):
            os.unlink(bitcode_modulename)

        main(args=['--llvm', '-o', bitcode_modulename, modulename + '.py'])

        # Sanity check bitcode file contents
        with open(bitcode_modulename, "rb") as f:
            bc = f.read()

        bitcode_wrapper_magic = b'\xde\xc0\x17\x0b'
        bitcode_magic = b'BC\xc0\xde'
        self.assertTrue(bc.startswith((bitcode_magic, bitcode_wrapper_magic)), bc)


if __name__ == "__main__":
    unittest.main()
