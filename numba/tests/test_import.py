import unittest
from unittest import mock
from numba.tests.support import TestCase, run_in_subprocess
from numba.core import utils
import os


class TestNumbaImport(TestCase):
    """
    Test behaviour of importing Numba.
    """

    def test_laziness(self):
        """
        Importing top-level numba features should not import too many modules.
        """
        # A heuristic set of modules that shouldn't be imported immediately
        banlist = ['cffi',
                   'distutils',
                   'numba.cuda',
                   'numba.cpython.mathimpl',
                   'numba.cpython.randomimpl',
                   'numba.tests',
                   'numba.core.typing.collections',
                   'numba.core.typing.listdecl',
                   'numba.np.types.datetime_registry',
                   ]

        # Sanity check the modules still exist...
        for mod in banlist:
            distutils_check = (mod != 'distutils' or
                               utils.PYVERSION < (3, 12))
            if mod not in ('cffi',) and distutils_check:
                __import__(mod)

        code = """if 1:
            from numba import jit, vectorize
            from numba.core import types
            import sys
            print(list(sys.modules))
            """

        out, _ = run_in_subprocess(code)
        modlist = set(eval(out.strip()))
        unexpected = set(banlist) & set(modlist)
        self.assertFalse(unexpected, "some modules unexpectedly imported")

    def test_no_impl_import(self):
        """
        Tests that importing jit does not trigger import of modules containing
        lowering implementations that would likely install things in the
        builtins registry and have side effects impacting other targets
        """
        # None of these modules should be imported through the process of
        # doing 'import numba' or 'from numba import njit'
        banlist = ['numba.cpython.slicing',
                   'numba.cpython.tupleobj',
                   'numba.cpython.enumimpl',
                   'numba.cpython.hashing',
                   'numba.cpython.heapq',
                   'numba.cpython.iterators',
                   'numba.cpython.numbers',
                   'numba.cpython.rangeobj',
                   'numba.cpython.cmathimpl',
                   'numba.cpython.mathimpl',
                   'numba.cpython.printimpl',
                   'numba.cpython.randomimpl',
                   'numba.core.optional',
                   'numba.misc.gdb_hook',
                   'numba.misc.literal',
                   'numba.misc.cffiimpl',
                   'numba.np.linalg',
                   'numba.np.polynomial',
                   'numba.np.arraymath',
                   'numba.np.npdatetime',
                   'numba.np.npyimpl',
                   'numba.typed.typeddict',
                   'numba.typed.typedlist',
                   'numba.experimental.jitclass.base',]

        code1 = """if 1:
            import sys
            import numba
            print(list(sys.modules))
            """

        code2 = """if 1:
            import sys
            from numba import njit
            @njit
            def foo():
                pass
            print(list(sys.modules))
            """

        for code in (code1, code2):
            out, _ = run_in_subprocess(code)
            modlist = set(eval(out.strip()))
            unexpected = set(banlist) & set(modlist)
            self.assertFalse(unexpected, "some modules unexpectedly imported")

    def test_no_accidental_warnings(self):
        # checks that importing Numba isn't accidentally triggering warnings due
        # to e.g. deprecated use of import locations from Python's stdlib
        code = "import numba"
        # See: https://github.com/numba/numba/issues/6831
        # bug in setuptools/packaging causing a deprecation warning
        flags = ["-Werror", "-Wignore::DeprecationWarning:packaging.version:"]
        run_in_subprocess(code, flags, env=os.environ.copy())

    def test_import_star(self):
        # checks that "from numba import *" works.
        code = "from numba import *"
        run_in_subprocess(code)


class TestEnsureCriticalDeps(TestCase):
    def test_numpy_min(self):
        import numpy as np
        import numba
        from numpy.lib import NumpyVersion
        floor = numba._ensure_critical_deps.min_numpy_version
        below = NumpyVersion(
            f"{floor.major}.{floor.minor}.{floor.bugfix - 1}")
        with mock.patch.object(np, '__version__', below.vstring):
            with self.assertRaises(ImportError):
                numba._ensure_critical_deps()
        with mock.patch.object(np, '__version__', floor.version):
            numba._ensure_critical_deps()
        with mock.patch.object(np, '__version__',
                               f"{floor.major}.{floor.minor}.10"):
            numba._ensure_critical_deps()
        with mock.patch.object(np, '__version__', f"{floor.major}.9.9"):
            with self.assertRaises(ImportError):
                numba._ensure_critical_deps()
        above_rc = f"{floor.major + 1}.0.0rc1"
        with mock.patch.object(np, '__version__', above_rc):
            numba._ensure_critical_deps()

    def test_scipy_min(self):
        import numba
        try:
            import scipy
        except ImportError:
            self.skipTest('scipy')
        with mock.patch.object(scipy, '__version__', '0.19.1'):
            with self.assertRaises(ImportError):
                numba._ensure_critical_deps()
        with mock.patch.object(scipy, '__version__', '1.0.0rc1'):
            numba._ensure_critical_deps()


if __name__ == '__main__':
    unittest.main()
