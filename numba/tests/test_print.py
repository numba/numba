import unittest

from numba.minivect import minitypes
from numba import *

@autojit(backend='ast')
def print_(value):
    print value

class TestPrint(unittest.TestCase):
    def test_print(self):
        # Test for no segfault :)
        print_(10)
        print_(10.0)
        print_("hello!")

if __name__ == "__main__":
    unittest.main()