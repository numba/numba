import unittest

from numba.minivect import minitypes
from numba import *

some_global = "hello"

@autojit(backend='ast')
def access_global():
    return some_global

@autojit(backend='ast')
def call_abs(num):
    return abs(num)

class TestConversion(unittest.TestCase):
    def test_globals(self):
        result = access_global()
        assert result == some_global, result

    def test_builtins(self):
        result = call_abs(-10)
        assert result == 10, result

if __name__ == "__main__":
    unittest.main()