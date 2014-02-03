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
    raise SpecialException("hello")

# ______________________________________________________________________

class TestRaise(unittest.TestCase):

    def _assert_raises(self, func, expected_args):
        try:
            func()
        except SpecialException as e:
            assert e.args == tuple(expected_args), (e.args, expected_args)
        else:
            raise AssertionError("Expected exception")

    def test_raise(self):
        self._assert_raises(raise1, [])
        self._assert_raises(raise2, ["hello"])


if __name__ == "__main__":
    unittest.main()
