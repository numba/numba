#!/usr/bin/env python

import sys
import numbapro

sys.exit(0 if numbapro.test(sys.argv[1:]) == 0 else 1)
