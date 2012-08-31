#! /usr/bin/env python
# ______________________________________________________________________

from numba import int32
from numba.decorators import jit

import unittest
import __builtin__

# ______________________________________________________________________

def int_pow_fn (val, exp):
    return val ** exp

# ______________________________________________________________________

def _int_pow (val, exp):
    x = 1
    temp = val
    w = exp
    while w > 0:
        if (w & 1) != 0:
            x = x * temp
            # TODO: Overflow check on x
        w >>= 1
        # Can save a multiply by doing a check on w, but break is not
        # currently supported...
        #if w == 0: break
        temp = temp * temp
        # TODO: Overflow check on temp
    return x

# ______________________________________________________________________

def bad_return_fn (arg0, arg1):
    arg0 + arg1

# ______________________________________________________________________

class TestIssues (unittest.TestCase):
    @unittest.skipUnless(hasattr(__builtin__, '__noskip__'),
                         "Having problem with @llvm.powi intrinsic.")
    def test_int_pow_fn (self):
        compiled_fn = jit(arg_types = (int32, int32), ret_type = int32)(
            int_pow_fn)
        self.assertEqual(compiled_fn(2, 3), 8)
        self.assertEqual(compiled_fn(3, 4), int_pow_fn(3, 4))

    def test_bad_return_fn (self):
        jit(arg_types = (int32, int32), ret_type = int32)(bad_return_fn)(0, 0)

# ______________________________________________________________________

if __name__ == '__main__':
    unittest.main()

# ______________________________________________________________________
# End of test_issues.py
