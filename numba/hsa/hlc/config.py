from __future__ import print_function, absolute_import

import os

os.environ['HSAILBIN'] = os.environ.get('HSAILBIN', '/opt/amd/bin')

DEFAULT_BUILTIN_PATH = "{0}/builtins-hsail.opt.bc".format(
    os.environ['HSAILBIN'])

BUILTIN_PATH = os.environ.get("NUMBA_HSAIL_BUILTINS_BC", DEFAULT_BUILTIN_PATH)
