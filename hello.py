from __future__ import print_function
import numpy
from numba import jit, int32, float64


@jit((float64[:], float64), nopython=True)
def foo(a, b):
    c = 0
    for i in range(a.shape[0]):
        c += a[i] + b
        a[i] += b
    return c

foo.jit((int32[:], int32), nopython=True)

print(foo.inspect_types())


@jit((int32,))
def bar(a):
    return str(a) + " is a number"


def main():
    a = numpy.arange(100, dtype='int32')
    b = 2
    c = foo(a, b)
    print(a)
    print(c)


    a = numpy.arange(100, dtype='float64')
    b = 2.
    c = foo(a, b)
    print(a)
    print(c)

    print(bar(2))



if __name__ == '__main__':
    main()
