import numpy as np
import numexpr as ne
from numba import *
from numbapro.vectorize.basic import BasicVectorize
from numbapro.vectorize.stream import StreamVectorize
from numbapro.vectorize.parallel import ParallelVectorize
from time import time
from math import sin

REPEAT = 50
CHECK_ERROR = False

def polynomial(a, b):
    # return a*a + b*b + 2*a*b + a*a*a + b*b*b + 5*a*a*b * 10*a*b*b
    # return a * a + b * b + 2 * a * b
    return a*a + b*b + 2*a*b + a*a*a + b*b*b + 5*a*a*b * 10*sin(a*b/(a*b))

def fix_time_unit(t):
    return '%.2f %s' % (t * 1e3, 'ms')

class Benchmark:
    def __init__(self):
        # build python ufunc for golden reference
        self.np_ufunc = np.vectorize(polynomial)

        self.dataA = np.linspace(1., 10000., 2**16)
        self.dataB = np.linspace(1., 10000., 2**16)

        self.golden = self.np_ufunc(self.dataA, self.dataB)

    def main(self):
        for ty in [np.float32, np.float64]:
            print(str(ty).center(80, '-'))
            ## measure against itself
            #self.print_evaluate(self.np_ufunc, ty, 'numpy')

            self.print_evaluate(self.build_numexpr(), ty, 'numexpr')

            self.print_evaluate(self.build_basic_vectorize(), ty,
                                'numba basic vectorize')

            self.print_evaluate(self.build_stream_vectorize(), ty,
                                'numba stream vectorize')

            self.print_evaluate(self.build_parallel_vectorize(), ty,
                                'numba parallel vectorize')

    def print_evaluate(self, func, ty, name):
        print('evaluate %s' % name)
        print('fastest = %s | average = %s | slowest = %s'
              % tuple(map(fix_time_unit, self.evaluate(func, ty))))

    def evaluate(self, func, ty):
        dataA = self.dataA.astype(ty) # cast
        dataB = self.dataB.astype(ty) # cast

        times = []
        for _ in range(REPEAT):
            ts = time()
            result = func(dataA, dataB) # run
            times.append(time() - ts)

            # check error
            if CHECK_ERROR:
                acceptable_error = 1e-6
                for i, (expect, got) in enumerate(zip(self.golden, result)):

                    error = abs(expect - got)/(expect + 1e-30)

                    if error > acceptable_error:
                        msg = 'error at i=%d | expect = %s | got = %s | error = %s'
                        raise ValueError(msg % (i, expect, got, error))

        return np.min(times), np.average(times), np.max(times)

    def build_basic_vectorize(self):
        pv = BasicVectorize(polynomial)
        # pv.add(ret_type=int32, arg_types=[int32, int32])
        pv.add(ret_type=f, arg_types=[f, f])
        pv.add(ret_type=d, arg_types=[d, d])
        ufunc = pv.build_ufunc()
        return ufunc

    def build_parallel_vectorize(self):
        pv = ParallelVectorize(polynomial)
        # pv.add(ret_type=int32, arg_types=[int32, int32])
        pv.add(ret_type=f, arg_types=[f, f])
        pv.add(ret_type=d, arg_types=[d, d])
        ufunc = pv.build_ufunc()
        return ufunc

    def build_stream_vectorize(self):
        pv = StreamVectorize(polynomial)
        # pv.add(ret_type=int32, arg_types=[int32, int32])
        pv.add(ret_type=f, arg_types=[f, f])
        pv.add(ret_type=d, arg_types=[d, d])
        ufunc = pv.build_ufunc(granularity=32)
        return ufunc

    def build_numexpr(self):
        # ne.set_num_threads(2)
        def func(a, b):
            # return ne.evaluate("a**2 + b**2 + 2*a*b + a**3 + b**3 + 5*a**2*b * 10*a*b**2")
            return ne.evaluate("a*a + b*b + 2*a*b + a*a*a + b*b*b + 5*a*a*b * 10*sin(a*b/(a*b))")

        return func


def _():


    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.linspace(0., 10000., 100).astype(ty)

        ts = time()
        result = para_ufunc(data, data)
        tnumba = time() - ts

        ts = time()
        gold = np_ufunc(data, data)
        tnumpy = time() - ts

        print("Numpy time: %fs" % tnumpy)
        print("Numba time: %fs" % tnumba)

        if tnumba < tnumpy:
            print("Numba is FASTER by %fx" % (tnumpy/tnumba))
        else:
            print("Numba is SLOWER by %fx" % (tnumba/tnumpy))



    test(np.double)
    test(np.float32)
    test(np.int32)


    print('All good')

if __name__ == '__main__':
    Benchmark().main()

