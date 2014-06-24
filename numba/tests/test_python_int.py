from __future__ import print_function
import numba.unittest_support as unittest
from numba import jit


class TestPythonInt(unittest.TestCase):
    def test_int_return_type(self):
        # Issue 474
        def f():
            return 5

        c_f = jit()(f)
        self.assertEqual(type(c_f()), type(f()))

if __name__ == '__main__':
    unittest.main()
