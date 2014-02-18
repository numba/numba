from __future__ import print_function, division, absolute_import
import sys
import os
import warnings

def _readenv(name, ctor, default):
    try:
        res = os.environ[name]
    except KeyError:
        return default
    else:
        try:
            return ctor(res)
        except:
            warnings.warn("environ %s defined but failed to parse '%s'" %
                          (name, res), RuntimeWarning)
            return default


# Debug flag to control compiler debug print
DEBUG = _readenv("NUMBA_DEBUG", int, 0)

# JIT Debug flag to trigger IR instruction print
DEBUG_JIT = _readenv("NUMBA_DEBUG_JIT", int, 0)

# Optimization level
OPT = _readenv("NUMBA_OPT", int, 0)

# Force dump of LLVM IR
DUMP_LLVM = _readenv("NUMBA_DUMP_LLVM", int, DEBUG)

# Force dump of Optimized LLVM IR
DUMP_OPTIMIZED = _readenv("NUMBA_DUMP_OPTIMIZED", int, DEBUG)

# Force dump of generated assembly
DUMP_ASSEMBLY = _readenv("NUMBA_DUMP_ASSEMBLY", int, DEBUG)

# Force dump of type annotation
ANNOTATE = _readenv("NUMBA_DUMP_ANNOTATION", int, 0)

# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]

