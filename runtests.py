#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys
import numba
sys.exit(0 if numba.test() else 1)
