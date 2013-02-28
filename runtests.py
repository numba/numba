#!/usr/bin/env python

import sys
import numba

sys.exit(0 if numba.test(sys.argv[1:]) == 0 else 1)
