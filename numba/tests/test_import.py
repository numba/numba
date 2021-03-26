import unittest
import subprocess
import sys

from numba.tests.support import TestCase


class TestNumbaImport(TestCase):
    """
    Test behaviour of importing Numba.
    """

    def run_in_subproc(self, code, flags=None):
        if flags is None:
            flags = []
        cmd = [sys.executable,] + flags + ["-c", code]
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            msg = "process failed with code %s: stderr follows\n%s\n"
            raise AssertionError(msg % (popen.returncode, err.decode()))
        return out, err

    def test_laziness(self):
        """
        Importing top-level numba features should not import too many modules.
        """
        # A heuristic set of modules that shouldn't be imported immediately
        blacklist = ['cffi',
                     'distutils',
                     'numba.cuda',
                     'numba.cpython.mathimpl',
                     'numba.cpython.randomimpl',
                     'numba.tests',
                     'numba.core.typing.collections',
                     'numba.core.typing.listdecl',
                     'numba.core.typing.npdatetime',
                     ]
        # Sanity check the modules still exist...
        for mod in blacklist:
            if mod not in ('cffi',):
                __import__(mod)

        code = """if 1:
            from numba import jit, vectorize
            from numba.core import types
            import sys
            print(list(sys.modules))
            """

        out, _ = self.run_in_subproc(code)
        modlist = set(eval(out.strip()))
        unexpected = set(blacklist) & set(modlist)
        self.assertFalse(unexpected, "some modules unexpectedly imported")

    def test_no_accidental_warnings(self):
        # checks that importing Numba isn't accidentally triggering warnings due
        # to e.g. deprecated use of import locations from Python's stdlib
        code = "import numba"
        # See: https://github.com/numba/numba/issues/6831
        # bug in setuptools/packaging causing a deprecation warning
        flags = ["-Werror", "-Wignore::DeprecationWarning:packaging.version:"]
        self.run_in_subproc(code, flags)

    def test_import_star(self):
        # checks that "from numba import *" works.
        code = "from numba import *"
        self.run_in_subproc(code)


if __name__ == '__main__':
    unittest.main()
