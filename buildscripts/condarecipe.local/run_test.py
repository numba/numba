import os
import numba.testing
if int(os.environ.get("NUMBA_MULTITEST", 1)):
    testfn = numba.testing.multitest
else:
    testfn = numba.testing.test
if not testfn():
    raise RuntimeError("Test failed")
print('numba.__version__: %s' % numba.__version__)
