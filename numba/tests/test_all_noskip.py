#! /usr/bin/env python
# ______________________________________________________________________
'''test_all

Grand unified unit test script for Numba.
'''
# ______________________________________________________________________

import __builtin__
__builtin__.__noskip__ = True

from test_all import *
from numba.tests import test_support

# ______________________________________________________________________

if __name__ == "__main__":
    test_support.main()

# ______________________________________________________________________
# End of test_all_noskip.py
