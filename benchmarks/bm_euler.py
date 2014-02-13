# Modified from a stackoverflow post by Hyperboreus:
# http://stackoverflow.com/questions/6964392/speed-comparison-with-project-euler-c-vs-python-vs-erlang-vs-haskell
from __future__ import print_function, division, absolute_import
import math
from numba import jit
from numba.utils import benchmark


def py_factorCount(n):
    square = math.sqrt(n)
    isquare = int (square)
    count = -1 if isquare == square else 0
    for candidate in range(1, isquare + 1):
        if not n % candidate:
            count += 2
    return count


def py_euler():
    triangle = 1
    index = 1
    while py_factorCount(triangle) < 1001:
        index += 1
        triangle += index
    return triangle


@jit("intp(intp)", nopython=True)
def factorCount(n):
    square = math.sqrt(n)
    isquare = int (square)
    count = -1 if isquare == square else 0
    for candidate in range(1, isquare + 1):
        if not n % candidate:
            count += 2
    return count


@jit("intp()", nopython=True)
def euler():
    triangle = 1
    index = 1
    while factorCount(triangle) < 1001:
        index += 1
        triangle += index
    return triangle

answer = 842161320


def numba_main():
    result = euler()
    assert result == answer


def python_main():
    result = py_euler()
    assert result == answer


if __name__ == '__main__':
    print(benchmark(python_main))
    print(benchmark(numba_main))
