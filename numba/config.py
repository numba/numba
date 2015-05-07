from __future__ import print_function, division, absolute_import

import struct
import sys
import os
import re
import warnings

import llvmlite.binding as ll


IS_WIN32 = sys.platform.startswith('win32')
MACHINE_BITS = tuple.__itemsize__ * 8
IS_32BITS = MACHINE_BITS == 32

_cpu_name = ll.get_host_cpu_name()


class NumbaWarning(Warning):
    pass


def _readenv(name, ctor, default):
    try:
        res = os.environ[name]
        if res == '':
            return default
    except KeyError:
        return default
    else:
        try:
            return ctor(res)
        except:
            warnings.warn("environ %s defined but failed to parse '%s'" %
                          (name, res), RuntimeWarning)
            return default

# Print warnings to screen about function compilation
#   0 = Numba warnings suppressed (default)
#   1 = All Numba warnings shown
WARNINGS = _readenv("NUMBA_WARNINGS", int, 0)
if WARNINGS == 0:
    warnings.simplefilter('ignore', NumbaWarning)

# Debug flag to control compiler debug print
DEBUG = _readenv("NUMBA_DEBUG", int, 0)

# JIT Debug flag to trigger IR instruction print
DEBUG_JIT = _readenv("NUMBA_DEBUG_JIT", int, 0)

# Enable debugging of front-end operation (up to and including IR generation)
DEBUG_FRONTEND = _readenv("NUMBA_DEBUG_FRONTEND", int, 0)

# Optimization level
OPT = _readenv("NUMBA_OPT", int, 3)

# Force dump of Python bytecode
DUMP_BYTECODE = _readenv("NUMBA_DUMP_BYTECODE", int, DEBUG_FRONTEND)

# Force dump of control flow graph
DUMP_CFG = _readenv("NUMBA_DUMP_CFG", int, DEBUG_FRONTEND)

# Force dump of Numba IR
DUMP_IR = _readenv("NUMBA_DUMP_IR", int, DEBUG_FRONTEND)

# Force dump of LLVM IR
DUMP_LLVM = _readenv("NUMBA_DUMP_LLVM", int, DEBUG)

# Force dump of Function optimized LLVM IR
DUMP_FUNC_OPT = _readenv("NUMBA_DUMP_FUNC_OPT", int, DEBUG)

# Force dump of Optimized LLVM IR
DUMP_OPTIMIZED = _readenv("NUMBA_DUMP_OPTIMIZED", int, DEBUG)

# Force disable loop vectorize
# Loop vectorizer is disabled on 32-bit win32 due to a bug (#649)
LOOP_VECTORIZE = _readenv("NUMBA_LOOP_VECTORIZE", int,
                          not (IS_WIN32 and IS_32BITS))

# Force dump of generated assembly
DUMP_ASSEMBLY = _readenv("NUMBA_DUMP_ASSEMBLY", int, DEBUG)

# Force dump of type annotation
ANNOTATE = _readenv("NUMBA_DUMP_ANNOTATION", int, 0)

# Dump type annotation in html format
HTML = _readenv("NUMBA_DUMP_HTML", str, None)

# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]

# Disable CUDA support
DISABLE_CUDA = _readenv("NUMBA_DISABLE_CUDA", int, 0)

# Allow interpreter fallback so that Numba @jit decorator will never fail
# Use for migrating from old numba (<0.12) which supported closure, and other
# yet-to-be-supported features.
COMPATIBILITY_MODE = _readenv("NUMBA_COMPATIBILITY_MODE", int, 0)

# Force CUDA compute capability
def _force_cc(text):
    if not text:
        return None
    else:
        m = re.match(r'(\d+)\.(\d+)', text)
        if not m:
            raise ValueError("NUMBA_FORCE_CUDA_CC must be specified as a "
                             "string of \"major.minor\" where major "
                             "and minor are decimals")
        grp = m.groups()
        return int(grp[0]), int(grp[1])


FORCE_CUDA_CC = _readenv("NUMBA_FORCE_CUDA_CC", _force_cc, None)

# x86-64 specific
# Enable AVX on supported platforms where it won't degrade performance.
ENABLE_AVX = _readenv("NUMBA_ENABLE_AVX", int,
                      _cpu_name not in ('corei7-avx', 'core-avx-i'))

# Disable jit for debugging
DISABLE_JIT = _readenv("NUMBA_DISABLE_JIT", int, 0)

# Enable CUDA simulator
ENABLE_CUDASIM = _readenv("NUMBA_ENABLE_CUDASIM", int, 0)
