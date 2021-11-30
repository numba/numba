import contextlib
import imp
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import skip
from ctypes import *

import numpy as np

import llvmlite.binding as ll

from numba.core import utils
from numba.pycc import main
from numba.pycc.decorators import clear_export_registry
from numba.pycc.platform import find_shared_ending, find_pyext_ending
from numba.pycc.platform import _external_compiler_ok

from numba.tests.support import (TestCase, tag, import_dynamic, temp_directory,
                                 has_blas, needs_external_compilers)
import unittest


try:
    import setuptools
except ImportError:
    setuptools = None

_skip_reason = 'windows only'
_windows_only = unittest.skipIf(not sys.platform.startswith('win'),
                                _skip_reason)


base_path = os.path.dirname(os.path.abspath(__file__))


def unset_macosx_deployment_target():
    """Unset MACOSX_DEPLOYMENT_TARGET because we are not building portable
    libraries
    """
    if 'MACOSX_DEPLOYMENT_TARGET' in os.environ:
        del os.environ['MACOSX_DEPLOYMENT_TARGET']

class TestCompilerChecks(TestCase):

    # NOTE: THIS TEST MUST ALWAYS RUN ON WINDOWS, DO NOT SKIP
    @_windows_only
    def test_windows_compiler_validity(self):
        # When inside conda-build VSINSTALLDIR should be set and windows should
        # have a valid compiler available, `_external_compiler_ok` should  agree
        # with this. If this is not the case then error out to alert devs.
        is_running_conda_build = os.environ.get('CONDA_BUILD', None) is not None
        if is_running_conda_build:
            if os.environ.get('VSINSTALLDIR', None) is not None:
                self.assertTrue(_external_compiler_ok)


class BasePYCCTest(TestCase):

    def setUp(self):
        unset_macosx_deployment_target()

        self.tmpdir = temp_directory('test_pycc')
        # Make sure temporary files and directories created by
        # distutils don't clutter the top-level /tmp
        tempfile.tempdir = self.tmpdir

    def tearDown(self):
        tempfile.tempdir = None
        # Since we're executing the module-under-test several times
        # from the same process, we must clear the exports registry
        # between invocations.
        clear_export_registry()

    @contextlib.contextmanager
    def check_c_ext(self, extdir, name):
        sys.path.append(extdir)
        try:
            lib = import_dynamic(name)
            yield lib
        finally:
            sys.path.remove(extdir)
            sys.modules.pop(name, None)


@needs_external_compilers
class TestLegacyAPI(BasePYCCTest):

    def test_pycc_ctypes_lib(self):
        """
        Test creating a C shared library object using pycc.
        """
        source = os.path.join(base_path, 'compile_with_pycc.py')
        cdll_modulename = 'test_dll_legacy' + find_shared_ending()
        cdll_path = os.path.join(self.tmpdir, cdll_modulename)
        if os.path.exists(cdll_path):
            os.unlink(cdll_path)

        main(args=['--debug', '-o', cdll_path, source])
        lib = CDLL(cdll_path)
        lib.mult.argtypes = [POINTER(c_double), c_void_p,
                             c_double, c_double]
        lib.mult.restype = c_int

        lib.multf.argtypes = [POINTER(c_float), c_void_p,
                              c_float, c_float]
        lib.multf.restype = c_int

        res = c_double()
        lib.mult(byref(res), None, 123, 321)
        self.assertEqual(res.value, 123 * 321)

        res = c_float()
        lib.multf(byref(res), None, 987, 321)
        self.assertEqual(res.value, 987 * 321)

    def test_pycc_pymodule(self):
        """
        Test creating a CPython extension module using pycc.
        """
        self.skipTest("lack of environment can make the extension crash")

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


@needs_external_compilers
class TestCC(BasePYCCTest):

    def setUp(self):
        super(TestCC, self).setUp()
        from numba.tests import compile_with_pycc
        self._test_module = compile_with_pycc
        imp.reload(self._test_module)

    @contextlib.contextmanager
    def check_cc_compiled(self, cc):
        #cc.verbose = True
        cc.output_dir = self.tmpdir
        cc.compile()

        with self.check_c_ext(self.tmpdir, cc.name) as lib:
            yield lib

    def check_cc_compiled_in_subprocess(self, lib, code):
        prolog = """if 1:
            import sys
            import types
            # to disable numba package
            sys.modules['numba'] = types.ModuleType('numba')
            try:
                from numba import njit
            except ImportError:
                pass
            else:
                raise RuntimeError('cannot disable numba package')

            sys.path.insert(0, %(path)r)
            import %(name)s as lib
            """ % {'name': lib.__name__,
                   'path': os.path.dirname(lib.__file__)}
        code = prolog.strip(' ') + code
        subprocess.check_call([sys.executable, '-c', code])

    def test_cc_properties(self):
        cc = self._test_module.cc
        self.assertEqual(cc.name, 'pycc_test_simple')

        # Inferred output directory
        d = self._test_module.cc.output_dir
        self.assertTrue(os.path.isdir(d), d)

        # Inferred output filename
        f = self._test_module.cc.output_file
        self.assertFalse(os.path.exists(f), f)
        self.assertTrue(os.path.basename(f).startswith('pycc_test_simple.'), f)
        if sys.platform.startswith('linux'):
            self.assertTrue(f.endswith('.so'), f)
            self.assertIn('.cpython', f)

    def test_compile(self):
        with self.check_cc_compiled(self._test_module.cc) as lib:
            res = lib.multi(123, 321)
            self.assertPreciseEqual(res, 123 * 321)
            res = lib.multf(987, 321)
            self.assertPreciseEqual(res, 987.0 * 321.0)
            res = lib.square(5)
            self.assertPreciseEqual(res, 25)
            self.assertIs(lib.get_none(), None)
            with self.assertRaises(ZeroDivisionError):
                lib.div(1, 0)

    def check_compile_for_cpu(self, cpu_name):
        cc = self._test_module.cc
        cc.target_cpu = cpu_name

        with self.check_cc_compiled(cc) as lib:
            res = lib.multi(123, 321)
            self.assertPreciseEqual(res, 123 * 321)
            self.assertEqual(lib.multi.__module__, 'pycc_test_simple')

    def test_compile_for_cpu(self):
        # Compiling for the host CPU should always succeed
        self.check_compile_for_cpu(ll.get_host_cpu_name())

    def test_compile_for_cpu_host(self):
        # Compiling for the host CPU should always succeed
        self.check_compile_for_cpu("host")

    @unittest.skipIf(sys.platform == 'darwin' and
                     utils.PYVERSION in ((3, 8), (3, 7)),
                     'distutils incorrectly using gcc on python 3.8 builds')
    def test_compile_helperlib(self):
        with self.check_cc_compiled(self._test_module.cc_helperlib) as lib:
            res = lib.power(2, 7)
            self.assertPreciseEqual(res, 128)
            for val in (-1, -1 + 0j, np.complex128(-1)):
                res = lib.sqrt(val)
                self.assertPreciseEqual(res, 1j)
            for val in (4, 4.0, np.float64(4)):
                res = lib.np_sqrt(val)
                self.assertPreciseEqual(res, 2.0)
            res = lib.spacing(1.0)
            self.assertPreciseEqual(res, 2**-52)
            # Implicit seeding at startup should guarantee a non-pathological
            # start state.
            self.assertNotEqual(lib.random(-1), lib.random(-1))
            res = lib.random(42)
            expected = np.random.RandomState(42).random_sample()
            self.assertPreciseEqual(res, expected)
            res = lib.size(np.float64([0] * 3))
            self.assertPreciseEqual(res, 3)

            code = """if 1:
                from numpy.testing import assert_equal, assert_allclose
                res = lib.power(2, 7)
                assert res == 128
                res = lib.random(42)
                assert_allclose(res, %(expected)s)
                res = lib.spacing(1.0)
                assert_allclose(res, 2**-52)
                """ % {'expected': expected}
            self.check_cc_compiled_in_subprocess(lib, code)

    def test_compile_nrt(self):
        with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
            # Sanity check
            self.assertPreciseEqual(lib.zero_scalar(1), 0.0)
            res = lib.zeros(3)
            self.assertEqual(list(res), [0, 0, 0])
            if has_blas:
                res = lib.vector_dot(4)
                self.assertPreciseEqual(res, 30.0)
            # test argsort
            val = np.float64([2., 5., 1., 3., 4.])
            res = lib.np_argsort(val)
            expected = np.argsort(val)
            self.assertPreciseEqual(res, expected)

            code = """if 1:
                from numpy.testing import assert_equal
                from numpy import float64, argsort
                res = lib.zero_scalar(1)
                assert res == 0.0
                res = lib.zeros(3)
                assert list(res) == [0, 0, 0]
                if %(has_blas)s:
                    res = lib.vector_dot(4)
                    assert res == 30.0
                val = float64([2., 5., 1., 3., 4.])
                res = lib.np_argsort(val)
                expected = argsort(val)
                assert_equal(res, expected)
                """ % dict(has_blas=has_blas)
            self.check_cc_compiled_in_subprocess(lib, code)

    def test_hashing(self):
        with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
            res = lib.hash_literal_str_A()
            self.assertPreciseEqual(res, hash("A"))
            res = lib.hash_str("A")
            self.assertPreciseEqual(res, hash("A"))
            
            code = """if 1:
                from numpy.testing import assert_equal
                res = lib.hash_literal_str_A()
                assert_equal(res, hash("A"))
                res = lib.hash_str("A")
                assert_equal(res, hash("A"))
                """
            self.check_cc_compiled_in_subprocess(lib, code)

    def test_c_extension_usecase(self):
        # Test C-extensions
        with self.check_cc_compiled(self._test_module.cc_nrt) as lib:
            arr = np.arange(128, dtype=np.intp)
            got = lib.dict_usecase(arr)
            expect = arr * arr
            self.assertPreciseEqual(got, expect)


@needs_external_compilers
class TestDistutilsSupport(TestCase):

    def setUp(self):
        unset_macosx_deployment_target()

        # Copy the test project into a temp directory to avoid
        # keeping any build leftovers in the source tree
        self.tmpdir = temp_directory('test_pycc_distutils')
        source_dir = os.path.join(base_path, 'pycc_distutils_usecase')
        self.usecase_dir = os.path.join(self.tmpdir, 'work')
        shutil.copytree(source_dir, self.usecase_dir)

    def check_setup_py(self, setup_py_file):
        # Compute PYTHONPATH to ensure the child processes see this Numba
        import numba
        numba_path = os.path.abspath(os.path.dirname(
                                     os.path.dirname(numba.__file__)))
        env = dict(os.environ)
        if env.get('PYTHONPATH', ''):
            env['PYTHONPATH'] = numba_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = numba_path

        def run_python(args):
            p = subprocess.Popen([sys.executable] + args,
                                 cwd=self.usecase_dir,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 env=env)
            out, _ = p.communicate()
            rc = p.wait()
            if rc != 0:
                self.fail("python failed with the following output:\n%s"
                          % out.decode('utf-8', 'ignore'))

        run_python([setup_py_file, "build_ext", "--inplace"])
        code = """if 1:
            import pycc_compiled_module as lib
            assert lib.get_const() == 42
            res = lib.ones(3)
            assert list(res) == [1.0, 1.0, 1.0]
            """
        run_python(["-c", code])

    def check_setup_nested_py(self, setup_py_file):
        # Compute PYTHONPATH to ensure the child processes see this Numba
        import numba
        numba_path = os.path.abspath(os.path.dirname(
                                     os.path.dirname(numba.__file__)))
        env = dict(os.environ)
        if env.get('PYTHONPATH', ''):
            env['PYTHONPATH'] = numba_path + os.pathsep + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = numba_path

        def run_python(args):
            p = subprocess.Popen([sys.executable] + args,
                                 cwd=self.usecase_dir,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 env=env)
            out, _ = p.communicate()
            rc = p.wait()
            if rc != 0:
                self.fail("python failed with the following output:\n%s"
                          % out.decode('utf-8', 'ignore'))

        run_python([setup_py_file, "build_ext", "--inplace"])
        code = """if 1:
            import nested.pycc_compiled_module as lib
            assert lib.get_const() == 42
            res = lib.ones(3)
            assert list(res) == [1.0, 1.0, 1.0]
            """
        run_python(["-c", code])

    def test_setup_py_distutils(self):
        self.check_setup_py("setup_distutils.py")

    def test_setup_py_distutils_nested(self):
        self.check_setup_nested_py("setup_distutils_nested.py")

    @unittest.skipIf(setuptools is None, "test needs setuptools")
    def test_setup_py_setuptools(self):
        self.check_setup_py("setup_setuptools.py")

    @unittest.skipIf(setuptools is None, "test needs setuptools")
    def test_setup_py_setuptools_nested(self):
        self.check_setup_nested_py("setup_setuptools_nested.py")


if __name__ == "__main__":
    unittest.main()
