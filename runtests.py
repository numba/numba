#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import numba

# TODO: Use argparse
if '--loop' in sys.argv:
    whitelist = [arg for arg in sys.argv[1:] if arg != '--loop']
    while True:
        exit_status = numba.test(whitelist)
        if exit_status != 0:
            sys.exit(exit_status)
else:
    sys.exit(numba.test(sys.argv[1:]))
