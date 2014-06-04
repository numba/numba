import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
from numba.config import PYVERSION

use_python = False


class TestUFuncs(unittest.TestCase):

    def test_binary_ufunc(self):

        a = numbarray.arange(10)
        result = numbarray.add(a, a)
        expected = np.add(np.arange(10), np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(1, 1)
        expected = np.add(1, 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(a, 1)
        expected = np.add(np.arange(10), 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = numbarray.add(1, a)
        expected = np.add(1, np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))


if __name__ == '__main__':
    unittest.main()

