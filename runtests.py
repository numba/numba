#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys

import numba.testing as testing

result = testing.test()
sys.exit(0 if result else 1)
