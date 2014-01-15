from __future__ import print_function
import numba.unittest_support as unittest
from numba import compiler, types, utils, dispatcher


def overhead(x):
    return x


class TestWrapper(unittest.TestCase):
    def test_overhead(self):
        cr = compiler.compile_isolated(overhead, [types.int32])
        cfunc = cr.entry_point
        disp = dispatcher.Overloaded(overhead)
        disp.add_overload(cr)
        direct = disp.dispatcher


        x = 321

        def python():
            overhead(x)

        def pycfunc():
            cfunc(x)

        def dispatched():
            direct((types.int32,), (x,))

        def overloaded():
            disp(x)

        MAXCT = 100000
        print(utils.benchmark(python, maxct=MAXCT))
        print(utils.benchmark(pycfunc, maxct=MAXCT))
        print(utils.benchmark(dispatched, maxct=MAXCT))
        print(utils.benchmark(overloaded, maxct=MAXCT))





if __name__ == '__main__':
    unittest.main()