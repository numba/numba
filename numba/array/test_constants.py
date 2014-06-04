import numba.unittest_support as unittest
import numba.array as numbarray
import numpy as np
import random
from numba.config import PYVERSION

use_python = False


class TestConstants(unittest.TestCase):
    def test_float_constant(self):
        """
        Test a truncation problem of long float constants
        """
        x = 0.123456789123456789123456789  # 9 x 3 = 27 digits
        self.float_constant_test(x, use_python=True)
        self.float_constant_test(x, use_python=False)

        # Test random floats
        for _ in range(20):
            x = random.random()
            self.float_constant_test(x, use_python=True)
            self.float_constant_test(x, use_python=False)

    def float_constant_test(self, realk, use_python):
        a = numbarray.arange(10)
        result = a + realk
        expected = np.arange(10) + realk
        got = result.eval(use_python=use_python)
        self.assertTrue(np.all(got == expected))


if __name__ == '__main__':
    unittest.main()

