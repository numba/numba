from __future__ import print_function
import os
import coverage
import matplotlib
matplotlib.use('agg')  # suppress matplotlib display
coverage.process_startup()
print(os.environ['COVERAGE_PROCESS_START'])
