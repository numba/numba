# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import


from numba import njit, cfunc
from numba.six import exec_
from numba.utils import PY2

from .support import TestCase, unittest

unicode_name1 = u"""
def unicode_name1(ಠ_ರೃ, ಠਊಠ):
    return (ಠ_ರೃ) + (ಠਊಠ)
"""

unicode_name2 = u"""
def Ծ_Ծ(ಠ_ರೃ, ಠਊಠ):
    return (ಠ_ರೃ) + (ಠਊಠ)
"""


@unittest.skipIf(PY2, "unicode identifier not supported in python2")
class TestUnicodeNames(TestCase):
    def make_testcase(self, src, fname):
        glb = {}
        exec_(src, glb)
        fn = glb[fname]
        return fn

    def test_unicode_name1(self):
        fn = self.make_testcase(unicode_name1, 'unicode_name1')
        cfn = njit(fn)
        self.assertEqual(cfn(1, 2), 3)

    def test_unicode_name2(self):
        fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
        cfn = njit(fn)
        self.assertEqual(cfn(1, 2), 3)

    def test_cfunc(self):
        fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
        cfn = cfunc("int32(int32, int32)")(fn)
        self.assertEqual(cfn.ctypes(1, 2), 3)


if __name__ == '__main__':
    unittest.main()

