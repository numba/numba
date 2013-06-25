# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import traceback
import unittest

from numba import autojit

# ______________________________________________________________________
# Helpers

class SpecialException(Exception):
    pass

def make_tb():
    try:
        raise SpecialException # hee hee hee
    except:
        type, val, tb = sys.exc_info()
        return tb

# ______________________________________________________________________

@autojit
def raise1():
    raise SpecialException

@autojit
def raise2():
    raise SpecialException, "hello"

@autojit
def raise3():
    raise SpecialException("hello")

@autojit
def raise4():
    raise SpecialException, "traceback!", make_tb()

# ______________________________________________________________________

class TestRaise(unittest.TestCase):

    def _assert_raises(self, func, expected_args):
        try:
            func()
        except SpecialException, e:
            assert e.args == tuple(expected_args), (e.args, expected_args)
        else:
            raise AssertionError("Expected exception")

    def test_raise(self):
        self._assert_raises(raise1, [])
        self._assert_raises(raise2, ["hello"])
        self._assert_raises(raise3, ["hello"])
        self._assert_raises(raise4, ["traceback!"])


    def test_raise_tb(self):
        try:
            raise4()
        except SpecialException:
            formatted = traceback.format_exc()
            self.assertIn("hee hee hee", formatted)
        else:
            raise AssertionError("Expected exception")

if __name__ == "__main__":
    unittest.main()
