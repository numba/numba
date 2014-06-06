import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
from numba.config import PYVERSION

use_python = False


class TestOperators(unittest.TestCase):


    def test_binary_operator(self):

        a = numbarray.arange(10)
        result = a + a
        expected = np.arange(10) + np.arange(10)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = a + 1
        expected = np.add(np.arange(10), 1)

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))

        result = 1 + a
        expected = np.add(1, np.arange(10))

        self.assertTrue(np.all(result.eval(use_python=use_python) == expected))


    def test_inplace_operator(self):

        a = numbarray.arange(10)
        x = a + a
        a += 1
        x1 = a + a

        a1 = np.arange(10)
        expected = a1 + a1
        a1 += 1
        expected1 = a1 + a1

        self.assertTrue(np.all(x.eval(use_python=use_python) == expected))
        self.assertTrue(np.all(x1.eval(use_python=use_python) == expected1))

if __name__ == '__main__':
    unittest.main()

