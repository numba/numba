import sys
import numba
if not numba.test():
    print("Test failed")
    sys.exit(1)
print('numba.__version__: %s' % numba.__version__)
