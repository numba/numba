from __future__ import print_function

import numpy as np

import numba.unittest_support as unittest
from numba import compiler, types, utils
from numba.targets import registry


def overhead(x):
    return x


def array_overhead(x):
    x[0] = 1
    x[1] = 2


def add(x):
    return x + x + x + x + x


class TestWrapper(unittest.TestCase):
    def test_overhead(self):
        """
        This will show higher overhead due to unboxing in the native version.
        """
        cr = compiler.compile_isolated(overhead, [types.int32])
        cfunc = cr.entry_point
        disp = registry.CPUDispatcher(overhead)
        disp.add_overload(cr)

        x = 321

        def python():
            overhead(x)

        def pycfunc():
            cfunc(x)

        def overloaded():
            disp(x)

        print(overhead)
        print(utils.benchmark(python, maxsec=.5))
        print(utils.benchmark(pycfunc, maxsec=.5))
        print(utils.benchmark(overloaded, maxsec=.5))

    def test_array_overhead(self):
        """
        The time to set two array element seems to be more expensive than
        the overhead of the overloaded call.
        """
        cr = compiler.compile_isolated(array_overhead, [types.int32[::1]])
        cfunc = cr.entry_point
        disp = registry.CPUDispatcher(array_overhead)
        disp.add_overload(cr)

        self.assertEqual(cr.signature.args[0].layout, 'C')

        x = np.zeros(shape=2, dtype='int32')

        def python():
            array_overhead(x)

        def pycfunc():
            cfunc(x)

        def overloaded():
            disp(x)

        print(array_overhead)
        print(utils.benchmark(python, maxsec=.5))
        print(utils.benchmark(pycfunc, maxsec=.5))
        print(utils.benchmark(overloaded, maxsec=.5))


    def test_add(self):
        """
        This seems to be about the amount of work to balance out the overhead
        by the overloaded one
        """
        cr = compiler.compile_isolated(add, [types.int32])
        cfunc = cr.entry_point
        disp = registry.CPUDispatcher(add)
        disp.add_overload(cr)

        x = 321

        def python():
            add(x)

        def pycfunc():
            cfunc(x)

        def overloaded():
            disp(x)

        print(add)
        print(utils.benchmark(python, maxsec=.5))
        print(utils.benchmark(pycfunc, maxsec=.5))
        print(utils.benchmark(overloaded, maxsec=.5))



if __name__ == '__main__':
    unittest.main()
