from __future__ import print_function, absolute_import

import subprocess
import sys

from numba import unittest_support as unittest
from .support import TestCase


class TestNumbaImport(TestCase):
    """
    Test behaviour of importing Numba.
    """

    def test_laziness(self):
        """
        Importing top-level numba features should not import too many modules.
        """
        # A heuristic set of modules that shouldn't be imported immediately
        blacklist = [
            'cffi',
            'distutils',
            'numba.cuda',
            'numba.hsa',
            'numba.targets.mathimpl',
            'numba.targets.randomimpl',
            'numba.tests',
            'numba.typing.collections',
            'numba.typing.listdecl',
            'numba.typing.npdatetime',
            ]
        # Sanity check the modules still exist...
        for mod in blacklist:
            if mod not in ('cffi', 'numba.hsa'):
                __import__(mod)

        code = """if 1:
            from numba import jit, types, vectorize
            import sys
            print(list(sys.modules))
            """

        popen = subprocess.Popen([sys.executable, "-c", code],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = popen.communicate()
        if popen.returncode != 0:
            raise AssertionError("process failed with code %s: stderr follows\n%s\n"
                                 % (popen.returncode, err.decode()))

        modlist = set(eval(out.strip()))
        unexpected = set(blacklist) & set(modlist)
        self.assertFalse(unexpected, "some modules unexpectedly imported")
