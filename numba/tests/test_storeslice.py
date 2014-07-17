from __future__ import print_function
import numba.unittest_support as unittest
import numpy as np
from numba.compiler import compile_isolated, Flags
from numba import types


def setitem_slice(a, start, stop, step, scalar): 
    a[start:stop:step] = scalar


def usecase(obs, nPoints, B, sigB, A, sigA, M, sigM):
    center = nPoints / 2
    print(center)
    obs[0:center] = np.arange(center)
    obs[center] = 321
    obs[(center + 1):] = np.arange(nPoints - center - 1)


class TestStoreSlice(unittest.TestCase):
    def test_usecase(self):
        n = 10
        obs_got = np.zeros(n)
        obs_expected = obs_got.copy()

        flags = Flags()
        flags.set("enable_pyobject")
        cres = compile_isolated(usecase, (), flags=flags)
        cres.entry_point(obs_got, n, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        usecase(obs_expected, n, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0)

        print(obs_got, obs_expected)
        self.assertTrue(np.allclose(obs_got, obs_expected))

    def test_array_slice_setitem(self):
        n = 10
        a = np.arange(n)
        b = np.arange(n)
        cres = compile_isolated(setitem_slice, (types.int64[:], types.int64, types.int64, types.int64, types.int64))
        cres.entry_point(a, 2, 6, 1, 7)
        setitem_slice(b, 2, 6, 1, 7)
        print(a)
        print(b)
        self.assertTrue(np.allclose(a, b))

if __name__ == '__main__':
    unittest.main()

