from llvm.core import *
from llvm_cbuilder import *
from llvm_cbuilder import shortnames as C

import unittest

class TestCstrCollide(unittest.TestCase):
    def test_same_string(self):
        mod = Module.new(__name__)
        cb = CBuilder.new_function(mod, 'test_cstr_collide', C.void, [])

        a = cb.constant_string("hello")
        b = cb.constant_string("hello")
        self.assertEqual(a.value, b.value)

if __name__ == '__main__':
    unittest.main()
