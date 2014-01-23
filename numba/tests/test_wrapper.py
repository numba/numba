from __future__ import print_function
import numba.unittest_support as unittest
from numba import compiler, types, utils, dispatcher
import numpy


def overhead(x):
    return x


def array_overhead(x):
    pass


class TestWrapper(unittest.TestCase):
    def test_overhead(self):
        cr = compiler.compile_isolated(overhead, [types.int32])
        cfunc = cr.entry_point
        disp = dispatcher.Overloaded(overhead)
        disp.add_overload(cr)

        x = 321

        def python():
            overhead(x)

        def pycfunc():
            cfunc(x)

        def overloaded():
            disp(x)

        MAXCT = 100000
        print(utils.benchmark(python, maxsec=.5))
        print(utils.benchmark(pycfunc, maxsec=.5))
        print(utils.benchmark(overloaded, maxsec=.5))

    def test_array_overhead(self):
        cr = compiler.compile_isolated(array_overhead, [types.int32[::1]])
        cfunc = cr.entry_point
        disp = dispatcher.Overloaded(array_overhead)
        disp.add_overload(cr)

        self.assertEqual(cr.argtypes[0].layout, 'C')

        x = numpy.zeros(shape=1, dtype='int32')

        def python():
            overhead(x)

        def pycfunc():
            cfunc(x)

        def overloaded():
            disp(x)

        MAXCT = 100000
        print(utils.benchmark(python, maxsec=.5))
        print(utils.benchmark(pycfunc, maxsec=.5))
        print(utils.benchmark(overloaded, maxsec=.5))



if __name__ == '__main__':
    unittest.main()