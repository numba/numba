"""
Sample low-level HSA runtime example.
"""
from __future__ import print_function, division

import numpy as np
from numba import hsa



if __name__=='__main__':
    hsa.init()

    hsa.shut_down()

    return 0
