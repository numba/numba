from __future__ import print_function
import numpy
from numba import jit, int8, int16, int32, float64, complex128


@jit((float64[:], float64), nopython=True)
def foo(a, b):
    c = 0
    for i in range(a.shape[0]):
        c += a[i] + b
        a[i] += b
    return c

cfoo = foo.jit((int32[:], int32), nopython=True)

print(foo.inspect_types())


@jit((int32,))
def bar(a):
    return str(a) + " is a number"


class Haha: pass


@jit
def ambiguous(x):
    return x

ambiguous.jit((int16,))
ambiguous.jit((int8,))
ambiguous.jit((complex128,))
ambiguous.jit((float64,))


def main():
    a = numpy.arange(100, dtype='int32')
    b = 2
    c = cfoo(a, b)
    print(a)
    print(c)

    a = numpy.arange(100, dtype='float64')
    b = 2.
    foo.disable_compile()
    c = foo(a, b)
    print(foo.overloads.keys())
    print(a)
    print(c)

    print(bar(2))
    print(bar(Haha()))
    bar.inspect_types()

    print(ambiguous(numpy.int8(1)))
    print(ambiguous(numpy.array(1, dtype='int16')))
    print(ambiguous(numpy.float64(1)))
    print(ambiguous(numpy.complex128(1)))
    ambiguous(numpy.int32(1))

if __name__ == '__main__':
    main()
