from __future__ import print_function
import os
import tempfile
import sys
from ctypes import *
from numba import unittest_support as unittest
from numba.pycc import find_shared_ending, main

base_path = os.path.dirname(os.path.abspath(__file__))


def unset_macosx_deployment_target():
    """Unset MACOSX_DEPLOYMENT_TARGET because we are not building portable
    libraries
    """
    macosx_target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', None)
    if macosx_target is not None:
        del os.environ['MACOSX_DEPLOYMENT_TARGET']


@unittest.skipIf(sys.platform.startswith("win32"), "Skip win32 test for now")
class TestPYCC(unittest.TestCase):

    def test_pycc_ctypes_lib(self):
        unset_macosx_deployment_target()

        modulename = os.path.join(base_path, 'compile_with_pycc')
        cdll_modulename = modulename + find_shared_ending()
        if os.path.exists(cdll_modulename):
            os.unlink(cdll_modulename)

        main(args=[modulename + '.py'])
        lib = CDLL(cdll_modulename)

        try:
            lib.mult.argtypes = [POINTER(c_double), c_void_p, c_double,
                                 c_double]
            lib.mult.restype = c_int

            lib.multf.argtypes = [POINTER(c_float), c_void_p, c_float, c_float]
            lib.multf.restype = c_int

            res = c_double()
            lib.mult(byref(res), None, 123, 321)
            self.assertEqual(res.value, 123 * 321)

            res = c_float()
            lib.multf(byref(res), None, 987, 321)
            self.assertEqual(res.value, 987 * 321)
        finally:
            del lib
            if os.path.exists(cdll_modulename):
                os.unlink(cdll_modulename)

    def test_pycc_pymodule(self):
        unset_macosx_deployment_target()

        modulename = os.path.join(base_path, 'compile_with_pycc')
        tmpdir = tempfile.gettempdir()
        out_modulename = (os.path.join(tmpdir, 'compiled_with_pycc')
                          + find_shared_ending())
        main(args=['--python', '-o', out_modulename, modulename + '.py'])

        sys.path.append(tmpdir)
        try:
            import compiled_with_pycc as lib
            try:
                res = lib.mult(123, 321)
                assert res == 123 * 321

                res = lib.multf(987, 321)
                assert res == 987 * 321
            finally:
                del lib
        finally:
            if os.path.exists(out_modulename):
                os.unlink(out_modulename)

if __name__ == "__main__":
    unittest.main()
