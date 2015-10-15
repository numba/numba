#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys

import numba.testing as testing

if __name__ == "__main__":
    result = testing.run_tests(sys.argv)
    sys.exit(0 if result else 1)
