import math
from typing import NamedTuple, Tuple, Union

import numpy as np
from xnd import xnd

from numba import unittest_support as unittest
from numba import int32, uint32, float32, float64, jit
from numba.gumath import jit_xnd
from numba import vectorize

pi = math.pi

def sinc(x):
    if x == 0.0:
        return 1.0
    else:
        return math.sin(x * pi) / (pi * x)

def scaled_sinc(x, scale):
    if x == 0.0:
        return scale
    else:
        return scale * (math.sin(x * pi) / (pi * x))

def vector_add(a, b):
    return a + b

def xnd_range(n, dtype):
    return xnd(list(range(n)), type=f'{n} * {dtype}')

@jit_xnd([
    'N * K * int32, N * K * int32 -> N * K * int32',
    'N * K * complex64, N * K * complex64 -> N * K * complex64',
])
def guadd(a, b, c):
    """A generalized addition"""
    x, y = c.shape
    for i in range(x):
        for j in range(y):
            c[i, j] = a[i, j] + b[i, j]

def equals(a, b):
    return a == b

def add_multiple_args(a, b, c, d):
    return a + b + c + d

inffered_compilation_cases = {
    # from test_vectorize_decor.py
    sinc: [
        [[xnd_range(100, d)], [np.arange(100).astype(d)]] for d in ['float64', 'float32', 'int32', 'uint32']
    ],
    scaled_sinc: [
        [
            [xnd_range(100, 'float64'), xnd(3, type='uint32')],
            [np.arange(100, dtype=np.float64), np.uint32(3)]
        ]
    ],
    vector_add: [
        [[xnd_range(100, d)] * 2, [np.arange(100).astype(d)] * 2] for d in ['float64', 'float32', 'int32', 'uint32']
    ] + [
        # Non-contiguous dimension
        [
            [xnd_range(12, 'int32')[::2], xnd_range(6, 'int32')],
            [np.arange(12, dtype='int32')[::2], np.arange(6, dtype='int32')]
        ],
        # complex numbers
        [
            [xnd([x + 1j for x in range(12)])] * 2,
            [np.arange(12, dtype='complex64') + 1j] * 2,
        ]
    ],
    # equals: [
    #     [
    #         [xnd_range(10, 'int32'), xnd_range(10, 'int32')],
    #         [np.arange(10, dtype='int32'), np.arange(10, dtype='int32')]
    #     ]
    # ]
}


class TestJITXnd(unittest.TestCase):
    def test_inferred_compilation(self):
        for fn, possible_args in inffered_compilation_cases.items():
            gu_func = jit_xnd(fn)
            numpy_func = np.vectorize(fn)
            for (xnd_args, np_args) in possible_args:
                result = gu_func(*xnd_args)
                gold = numpy_func(*np_args)
                np.testing.assert_allclose(result, gold, atol=1e-8)

    def test_object_mode(self):
        @jit_xnd(forceobj=True)
        def t(a):
            return a
        with self.assertRaises(NotImplementedError):
            t(xnd(0))
    
    def test_nested_call(self):
        # TODO
        pass

    def test_nested_call_explicit_types(self):
        # TODO
        pass
    
    def test_multidimensional_add(self):
        a = np.arange(10, dtype="int32").reshape(2, 5)
        gold = a + a
        # since xnd doesn't have reshape yet
        b = xnd([list(range(5)), list(range(5, 10))], dtype='int32')
        result = guadd(b, b)
        np.testing.assert_allclose(result, gold)

        a = np.arange(10, dtype="complex64").reshape(2, 5) + 1j
        gold = a + a
        b = xnd([
            [i + 1j for i in range(5)],
            [i + 1j for i in range(5, 10)],
        ], dtype='complex64')
        result = guadd(b, b)
        np.testing.assert_allclose(result, gold)

    def test_only_kwargs(self):
        f = jit_xnd(nopython=True)(lambda a: a)
        np.testing.assert_allclose(f(xnd(0)), xnd(0))

    # def test_broadcasting(self):
    #     def test(args):

    #     args = 
    #     a = np.arange(80, dtype='float32').reshape(8, 10)
    #     b = a.copy()
    #     c = a.copy(order='F')
    #     d = np.arange(16 * 20, dtype='float32').reshape(16, 20)[::2, ::2]

    #     xnd_a = xnd([for ])


if __name__ == '__main__':
    unittest.main()
