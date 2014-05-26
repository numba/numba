from __future__ import print_function
import numba.unittest_support as unittest
from numba import jit


class TestClosure(unittest.TestCase):
    def test_jit_closure_variable(self):
        Y = 10

        def add_Y(x):
            return x + Y

        c_add_Y = jit('i4(i4)', nopython=True)(add_Y)
        self.assertEquals(c_add_Y(1), 11)

        # Like globals in Numba, the value of the closure is captured
        # at time of JIT
        Y = 12  # should not affect function
        self.assertEquals(c_add_Y(1), 11)

    def test_jit_multiple_closure_variables(self):
        Y = 10
        Z = 2

        def add_Y_mult_Z(x):
            return (x + Y) * Z

        c_add_Y_mult_Z = jit('i4(i4)', nopython=True)(add_Y_mult_Z)
        self.assertEquals(c_add_Y_mult_Z(1), 22)

    def test_jit_inner_function(self):
        @jit('i4(i4)', nopython=True)
        def mult_10(a):
            return a * 10

        def do_math(x):
            return mult_10(x + 4)

        c_do_math = jit('i4(i4)', nopython=True)(do_math)

        self.assertEquals(c_do_math(1), 50)


if __name__ == '__main__':
    unittest.main()
