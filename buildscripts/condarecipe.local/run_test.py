import sys
import numba

args = []
if sys.platform.startswith('win32'):
    args += ['-b']
else:
    args += ['-m', '-b']

if not numba.test(*args):
    print("Test failed")
    sys.exit(1)
print('numba.__version__: %s' % numba.__version__)
