from __future__ import print_function, absolute_import

import sys
import os

# Set a default for HSAILBIN if it is not defined.
# This is only used for cmdline HLC
os.environ['HSAILBIN'] = os.environ.get('HSAILBIN', '/opt/amd/bin')

# The default location of the HSAIL builtins library
DEFAULT_BUILTIN_PATH = os.path.join(sys.prefix, 'lib', 'builtins-hsail.opt.bc')

# The path where numba will look for the HSAIL builtins library.
# Use user specified path if it is defined.
BUILTIN_PATH = os.environ.get("NUMBA_HSAIL_BUILTINS_BC", DEFAULT_BUILTIN_PATH)
