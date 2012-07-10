import sys

try:
    from . import minivect
except ImportError:
    print >>sys.stderr, "Did you forget to update submodule minivect?"
    print >>sys.stderr, "Run 'git submodule init' followed by 'git submodule update'"
    raise

from . import _numba_types
from ._numba_types import *

__all__ = _numba_types.__all__
