import sys
import numba

if sys.platform.startswith('win32'):
    sys.argv += ['-b']
else:
    sys.argv += ['-m', '-b']

if not numba.test():
    print("Test failed")
    sys.exit(1)
print('numba.__version__: %s' % numba.__version__)
