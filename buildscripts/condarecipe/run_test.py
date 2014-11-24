import sys
import numba

sys.argv += ['-m', '-b']

if not numba.test():
    print("Test failed")
    sys.exit(1)
print('numba.__version__: %s' % numba.__version__)
