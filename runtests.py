#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
# -*- coding: utf-8 -*-

import sys
import numba

sys.exit(0 if numba.test(sys.argv[1:]) == 0 else 1)
