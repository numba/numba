"""
Expose top-level symbols that are safe for import *
"""
from __future__ import print_function, division, absolute_import
from . import testing, decorators
from ._version import get_versions
from . import special, types, config
# Re-export typeof
from .special import *
from .pycc.decorators import export, exportmany

# Version
__version__ = get_versions()['version']

# Re-export all type names
from .types import *

# Re export decorators
jit = decorators.jit
autojit = decorators.autojit
njit = decorators.njit

# Re export vectorize decorators
from .npyufunc import vectorize, guvectorize

# Re-export test entrypoint
test = testing.test

# Try initialize cuda
from . import cuda

__all__ = """
jit
autojit
njit
vectorize
guvectorize
export
exportmany
cuda
""".split() + types.__all__ + special.__all__

import llvm
# git builds append a dash and some other stuff after the version '0.12.6'
llvm_ver = llvm.__version__.split('-') 
llvm_ver = llvm_ver[0].split('.')
if (int(llvm_ver[0]), int(llvm_ver[1]), int(llvm_ver[2])) >= (0, 12, 6):
    pass
else:
    print("Numba requires at least version 0.12.6 of llvmpy.\nPlease update your version of llvmpy.")
    raise SystemExit(-1) 

