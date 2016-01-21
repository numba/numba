import sys
from numba.testing import run_tests

if __name__ == '__main__':
    sys.exit(0 if run_tests(sys.argv).wasSuccessful() else 1)
