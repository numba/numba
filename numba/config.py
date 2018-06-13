from __future__ import print_function, division, absolute_import

import platform
import struct
import sys
import os
import re
import warnings
import multiprocessing

# YAML needed to use file based Numba config
try:
    import yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False


import llvmlite.binding as ll

IS_WIN32 = sys.platform.startswith('win32')
IS_OSX = sys.platform.startswith('darwin')
MACHINE_BITS = tuple.__itemsize__ * 8
IS_32BITS = MACHINE_BITS == 32
# Python version in (major, minor) tuple
PYVERSION = sys.version_info[:2]

# this is the name of the user supplied configuration file
_config_fname = '.numba_config.yaml'

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

        # first check if there's a .numba_config.yaml and use values from that
        if os.path.exists(_config_fname) and os.path.isfile(_config_fname):
            if not _HAVE_YAML:
                msg = ("A Numba config file is found but YAML parsing "
                       "capabilities appear to be missing. "
                       "To use this feature please install `pyyaml`. e.g. "
                       "`conda install pyyaml`.")
                warnings.warn(msg)
            else:
                with open(_config_fname, 'rt') as f:
                    y_conf = yaml.load(f)
                if y_conf is not None:
                    for k, v in y_conf.items():
                        new_environ['NUMBA_' + k.upper()] = v

        # clobber file based config with any locally defined env vars
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
                              (name, value), RuntimeWarning)
                return default

        def optional_str(x):
            return str(x) if x is not None else None

        # Print warnings to screen about function compilation
        #   0 = Numba warnings suppressed (default)
        #   1 = All Numba warnings shown
        WARNINGS = _readenv("NUMBA_WARNINGS", int, 0)

        # developer mode produces full tracebacks, disables help instructions
        DEVELOPER_MODE = _readenv("NUMBA_DEVELOPER_MODE", int, 0)

        # Flag to enable full exception reporting
        FULL_TRACEBACKS = _readenv("NUMBA_FULL_TRACEBACKS", int, DEVELOPER_MODE)

        # Show help text when an error occurs
        SHOW_HELP = _readenv("NUMBA_SHOW_HELP", int, not DEVELOPER_MODE)

        # The color scheme to use for error messages, default is no color
        # just bold fonts in use.
        COLOR_SCHEME = _readenv("NUMBA_COLOR_SCHEME", str, "no_color")

        # Debug flag to control compiler debug print
        DEBUG = _readenv("NUMBA_DEBUG", int, 0)

        # JIT Debug flag to trigger IR instruction print
        DEBUG_JIT = _readenv("NUMBA_DEBUG_JIT", int, 0)

        # Enable debugging of front-end operation (up to and including IR generation)
        DEBUG_FRONTEND = _readenv("NUMBA_DEBUG_FRONTEND", int, 0)

        # Enable logging of cache operation
        DEBUG_CACHE = _readenv("NUMBA_DEBUG_CACHE", int, DEBUG)

        # Redirect cache directory
        # Contains path to the directory
        CACHE_DIR = _readenv("NUMBA_CACHE_DIR", str, "")

        # Enable tracing support
        TRACE = _readenv("NUMBA_TRACE", int, 0)

        # Enable debugging of type inference
        DEBUG_TYPEINFER = _readenv("NUMBA_DEBUG_TYPEINFER", int, 0)

        # Configure compilation target to use the specified CPU name
        # and CPU feature as the host information.
        # Note: this overrides "host" option for AOT compilation.
        CPU_NAME = _readenv("NUMBA_CPU_NAME", optional_str, None)
        CPU_FEATURES = _readenv("NUMBA_CPU_FEATURES", optional_str,
                                ("" if str(CPU_NAME).lower() == 'generic'
                                 else None))
        # Optimization level
        OPT = _readenv("NUMBA_OPT", int, 3)

        # Force dump of Python bytecode
        DUMP_BYTECODE = _readenv("NUMBA_DUMP_BYTECODE", int, DEBUG_FRONTEND)

        # Force dump of control flow graph
        DUMP_CFG = _readenv("NUMBA_DUMP_CFG", int, DEBUG_FRONTEND)

        # Force dump of Numba IR
        DUMP_IR = _readenv("NUMBA_DUMP_IR", int,
                           DEBUG_FRONTEND or DEBUG_TYPEINFER)

        # print debug info of analysis and optimization on array operations
        DEBUG_ARRAY_OPT = _readenv("NUMBA_DEBUG_ARRAY_OPT", int, 0)

        # insert debug stmts to print information at runtime
        DEBUG_ARRAY_OPT_RUNTIME = _readenv("NUMBA_DEBUG_ARRAY_OPT_RUNTIME", int, 0)

        # print stats about parallel for-loops
        DEBUG_ARRAY_OPT_STATS = _readenv("NUMBA_DEBUG_ARRAY_OPT_STATS", int, 0)

        # print debug info of inline closure pass
        DEBUG_INLINE_CLOSURE = _readenv("NUMBA_DEBUG_INLINE_CLOSURE", int, 0)

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

        # Dump IR in such as way as to aid in "diff"ing.
        DIFF_IR = _readenv("NUMBA_DIFF_IR", int, 0)

        # Dump type annotation in html format
        def fmt_html_path(path):
            if path is None:
                return path
            else:
                return os.path.abspath(path)

        HTML = _readenv("NUMBA_DUMP_HTML", fmt_html_path, None)

        # Allow interpreter fallback so that Numba @jit decorator will never fail
        # Use for migrating from old numba (<0.12) which supported closure, and other
        # yet-to-be-supported features.
        COMPATIBILITY_MODE = _readenv("NUMBA_COMPATIBILITY_MODE", int, 0)

        # x86-64 specific
        # Enable AVX on supported platforms where it won't degrade performance.
        def avx_default():
            if not _os_supports_avx():
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

        # if set and SVML is available, it will be disabled
        # By default, it's disabled on 32-bit platforms.
        DISABLE_INTEL_SVML = _readenv("NUMBA_DISABLE_INTEL_SVML", int, IS_32BITS)

        # Disable jit for debugging
        DISABLE_JIT = _readenv("NUMBA_DISABLE_JIT", int, 0)

        # CUDA Configs

        # Force CUDA compute capability to a specific version
        FORCE_CUDA_CC = _readenv("NUMBA_FORCE_CUDA_CC", _parse_cc, None)

        # Disable CUDA support
        DISABLE_CUDA = _readenv("NUMBA_DISABLE_CUDA", int, int(MACHINE_BITS==32))

        # Enable CUDA simulator
        ENABLE_CUDASIM = _readenv("NUMBA_ENABLE_CUDASIM", int, 0)

        # CUDA logging level
        # Any level name from the *logging* module.  Case insensitive.
        # Defaults to CRITICAL if not set or invalid.
        # Note: This setting only applies when logging is not configured.
        #       Any existing logging configuration is preserved.
        CUDA_LOG_LEVEL = _readenv("NUMBA_CUDA_LOG_LEVEL", str, '')

        # Maximum number of pending CUDA deallocations (default: 10)
        CUDA_DEALLOCS_COUNT = _readenv("NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT",
                                       int, 10)

        # Maximum ratio of pending CUDA deallocations to capacity (default: 0.2)
        CUDA_DEALLOCS_RATIO = _readenv("NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO",
                                       float, 0.2)

        # HSA Configs

        # Disable HSA support
        DISABLE_HSA = _readenv("NUMBA_DISABLE_HSA", int, 0)

        # The default number of threads to use.
        NUMBA_DEFAULT_NUM_THREADS = max(1, multiprocessing.cpu_count())

        # Numba thread pool size (defaults to number of CPUs on the system).
        NUMBA_NUM_THREADS = _readenv("NUMBA_NUM_THREADS", int,
                                     NUMBA_DEFAULT_NUM_THREADS)

        # Debug Info

        # The default value for the `debug` flag
        DEBUGINFO_DEFAULT = _readenv("NUMBA_DEBUGINFO", int, 0)
        CUDA_DEBUGINFO_DEFAULT = _readenv("NUMBA_CUDA_DEBUGINFO", int, 0)

        # Inject the configuration values into the module globals
        for name, value in locals().copy().items():
            if name.isupper():
                globals()[name] = value

        # delay this until now, let the globals for the module be updated
        # prior to loading numba.errors as it needs to use the config
        if WARNINGS == 0:
            from numba.errors import NumbaWarning
            warnings.simplefilter('ignore', NumbaWarning)
        if not _os_supports_avx():
            from numba.errors import PerformanceWarning
            warnings.warn("your operating system doesn't support "
                            "AVX, this may degrade performance on "
                            "some numerical code", PerformanceWarning)

_env_reloader = _EnvReloader()


def reload_config():
    """
    Reload the configuration from environment variables, if necessary.
    """
    _env_reloader.update()
