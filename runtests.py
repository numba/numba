#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys
import numba

# TODO: Use argparse
if '--loop' in sys.argv:
    whitelist = [arg for arg in sys.argv[1:] if arg != '--loop']
    sys.exit(numba.test(whitelist, loop=True))
else:
    sys.exit(numba.test(sys.argv[1:]))
