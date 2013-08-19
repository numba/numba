import sys
import numba

blacklist = ['test_ad']
if sys.platform == 'win32':
    blacklist.append('test_pycc_tresult')
    if tuple.__itemsize__ == 8:
        blacklist.append('test_numpy_math')
        blacklist.append('test_cstring')
if sys.platform == 'linux2' and tuple.__itemsize__ == 4:
    blacklist.append('test_ctypes_call')
    blacklist.append('test_ctypes')
    if sys.version_info[0] == 3:
        blacklist.append('test_issue_57')

try:
    import meta
except ImportError:
    blacklist.append('test_nosource')

if sys.platform == 'win32' and sys.version_info[0] == 3:
    print("*** Skipping tests ***")
else:
    assert numba.test(blacklist=blacklist) == 0

import os, sys
from numba import minivect

lst = os.listdir(minivect.get_include())
assert 'miniutils.h' in lst
assert 'miniutils.pyx' in lst


import mandel

print('numba.__version__: %s' % numba.__version__)
#assert numba.__version__ == '0.10.0'
