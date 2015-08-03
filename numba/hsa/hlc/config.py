from __future__ import print_function, absolute_import

import sys
import os

os.environ['HSAILBIN'] = os.environ.get('HSAILBIN', '/opt/amd/bin')

DEFAULT_BUILTIN_PATH = os.path.join(sys.prefix, 'lib', 'builtins-hsail.opt.bc')

BUILTIN_PATH = os.environ.get("NUMBA_HSAIL_BUILTINS_BC", DEFAULT_BUILTIN_PATH)
