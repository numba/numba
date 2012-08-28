#! /usr/bin/env python
# ______________________________________________________________________

from numba import int32
from numba.decorators import jit

import unittest

# ______________________________________________________________________

def int_pow_fn (val, exp):
    return val ** exp

# ______________________________________________________________________

def bad_return_fn (arg0, arg1):
    arg0 + arg1

# ______________________________________________________________________

class TestIssues (unittest.TestCase):
    def test_int_pow_fn (self):
        compiled_fn = jit(arg_types = (int32, int32), ret_type = int32)(
            int_pow_fn)
        self.assertEqual(compiled_fn(2, 3), 8)
        self.assertEqual(compiled_fn(3, 4), int_pow_fn(3, 4))

    def test_bad_return_fn (self):
        self.assertRaises(jit(arg_types = (int32, int32), ret_type = int32)(
                bad_return_fn),
                          Exception)

# ______________________________________________________________________

if __name__ == '__main__':
    unittest.main()

# ______________________________________________________________________
# End of test_issues.py
