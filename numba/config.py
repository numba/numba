from __future__ import print_function, division, absolute_import

import platform
import struct
import sys
import os
import re
import warnings

import llvmlite.binding as ll

from .errors import NumbaWarning, PerformanceWarning


IS_WIN32 = sys.platform.startswith('win32')
MACHINE_BITS = tuple.__itemsize__ * 8
IS_32BITS = MACHINE_BITS == 32
# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]


def _parse_cc(text):
    """
    Parse CUDA compute capability version string.
    """
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


def _os_supports_avx():
    """
    Whether the current OS supports AVX, regardless of the CPU.

    This is necessary because the user may be running a very old Linux
    kernel (e.g. CentOS 5) on a recent CPU.
    """
    if (not sys.platform.startswith('linux')
        or platform.machine() not in ('i386', 'i586', 'i686', 'x86_64')):
        return True
    # Executing the CPUID instruction may report AVX available even though
    # the kernel doesn't support it, so parse /proc/cpuinfo instead.
    try:
        f = open('/proc/cpuinfo', 'r')
    except OSError:
        # If /proc isn't available, assume yes
        return True
    with f:
        for line in f:
            head, _, body = line.partition(':')
            if head.strip() == 'flags' and 'avx' in body.split():
                return True
        else:
            return False


class _EnvReloader(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.old_environ = {}
        self.update(force=True)

    def update(self, force=False):
        new_environ = {}
        for name, value in os.environ.items():
            if name.startswith('NUMBA_'):
                new_environ[name] = value
        # We update the config variables if at least one NUMBA environment
        # variable was modified.  This lets the user modify values
        # directly in the config module without having them when
        # reload_config() is called by the compiler.
        if force or self.old_environ != new_environ:
            self.process_environ(new_environ)
            # Store a copy
            self.old_environ = dict(new_environ)

    def process_environ(self, environ):
        def _readenv(name, ctor, default):
            value = environ.get(name)
            if value is None:
                return default() if callable(default) else default
            try:
                return ctor(value)
            except Exception:
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

        # Enable logging of cache operation
        DEBUG_CACHE = _readenv("NUMBA_DEBUG_CACHE", int, DEBUG)

        # Enable tracing support
        TRACE = _readenv("NUMBA_TRACE", int, 0)

        # Enable debugging of type inference
        DEBUG_TYPEINFER = _readenv("NUMBA_DEBUG_TYPEINFER", int, 0)

        # Optimization level
        OPT = _readenv("NUMBA_OPT", int, 3)

        # Force dump of Python bytecode
        DUMP_BYTECODE = _readenv("NUMBA_DUMP_BYTECODE", int, DEBUG_FRONTEND)

        # Force dump of control flow graph
        DUMP_CFG = _readenv("NUMBA_DUMP_CFG", int, DEBUG_FRONTEND)

        # Force dump of Numba IR
        DUMP_IR = _readenv("NUMBA_DUMP_IR", int,
                           DEBUG_FRONTEND or DEBUG_TYPEINFER)

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

        # Disable CUDA support
        DISABLE_CUDA = _readenv("NUMBA_DISABLE_CUDA", int, int(MACHINE_BITS==32))

        # Allow interpreter fallback so that Numba @jit decorator will never fail
        # Use for migrating from old numba (<0.12) which supported closure, and other
        # yet-to-be-supported features.
        COMPATIBILITY_MODE = _readenv("NUMBA_COMPATIBILITY_MODE", int, 0)

        # Force CUDA compute capability to a specific version
        FORCE_CUDA_CC = _readenv("NUMBA_FORCE_CUDA_CC", _parse_cc, None)

        # x86-64 specific
        # Enable AVX on supported platforms where it won't degrade performance.
        def avx_default():
            if not _os_supports_avx():
                warnings.warn("your operating system doesn't support "
                              "AVX, this may degrade performance on "
                              "some numerical code", PerformanceWarning)
                return False
            else:
                # There are various performance issues with AVX and LLVM
                # on some CPUs (list at
                # http://llvm.org/bugs/buglist.cgi?quicksearch=avx).
                # For now we'd rather disable it, since it can pessimize the code.
                cpu_name = ll.get_host_cpu_name()
                return cpu_name not in ('corei7-avx', 'core-avx-i',
                                        'sandybridge', 'ivybridge')

        ENABLE_AVX = _readenv("NUMBA_ENABLE_AVX", int, avx_default)

        # Disable jit for debugging
        DISABLE_JIT = _readenv("NUMBA_DISABLE_JIT", int, 0)

        # Enable CUDA simulator
        ENABLE_CUDASIM = _readenv("NUMBA_ENABLE_CUDASIM", int, 0)

        # HSA Configs

        # Disable HSA support
        DISABLE_HSA = _readenv("NUMBA_DISABLE_HSA", int, 0)
        # Inject the configuration values into the module globals
        for name, value in locals().copy().items():
            if name.isupper():
                globals()[name] = value


_env_reloader = _EnvReloader()


def reload_config():
    """
    Reload the configuration from environment variables, if necessary.
    """
    _env_reloader.update()
