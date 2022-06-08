import platform
import sys
import os
import re
import shutil
import warnings

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
            raise ValueError("Compute capability must be specified as a "
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


# Choose how to handle captured errors
def _validate_captured_errors_style(style_str):
    rendered_style = str(style_str)
    if rendered_style not in ('new_style', 'old_style'):
        msg = ("Invalid style in NUMBA_CAPTURED_ERRORS: "
               f"{rendered_style}")
        raise ValueError(msg)
    else:
        return rendered_style


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
                    y_conf = yaml.safe_load(f)
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

        self.validate()

    def validate(self):
        global CUDA_USE_NVIDIA_BINDING

        if CUDA_USE_NVIDIA_BINDING:  # noqa: F821
            try:
                import cuda  # noqa: F401
            except ImportError as ie:
                msg = ("CUDA Python bindings requested (the environment "
                       "variable NUMBA_CUDA_USE_NVIDIA_BINDING is set), "
                       f"but they are not importable: {ie.msg}.")
                warnings.warn(msg)

                CUDA_USE_NVIDIA_BINDING = False

            if CUDA_PER_THREAD_DEFAULT_STREAM:  # noqa: F821
                warnings.warn("PTDS support is handled by CUDA Python when "
                              "using the NVIDIA binding. Please set the "
                              "environment variable "
                              "CUDA_PYTHON_CUDA_PER_THREAD_DEFAULT_STREAM to 1 "
                              "instead.")

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

        # developer mode produces full tracebacks, disables help instructions
        DEVELOPER_MODE = _readenv("NUMBA_DEVELOPER_MODE", int, 0)

        # disable performance warnings, will switch of the generation of
        # warnings of the class NumbaPerformanceWarning
        DISABLE_PERFORMANCE_WARNINGS = _readenv(
            "NUMBA_DISABLE_PERFORMANCE_WARNINGS", int, 0)

        # Flag to enable full exception reporting
        FULL_TRACEBACKS = _readenv(
            "NUMBA_FULL_TRACEBACKS", int, DEVELOPER_MODE)

        # Show help text when an error occurs
        SHOW_HELP = _readenv("NUMBA_SHOW_HELP", int, 0)

        # The color scheme to use for error messages, default is no color
        # just bold fonts in use.
        COLOR_SCHEME = _readenv("NUMBA_COLOR_SCHEME", str, "no_color")

        # Whether to globally enable bounds checking. The default None means
        # to use the value of the flag to @njit. 0 or 1 overrides the flag
        # globally.
        BOUNDSCHECK = _readenv("NUMBA_BOUNDSCHECK", int, None)

        # Whether to always warn about potential uninitialized variables
        # because static controlflow analysis cannot find a definition
        # in one or more of the incoming paths.
        ALWAYS_WARN_UNINIT_VAR = _readenv(
            "NUMBA_ALWAYS_WARN_UNINIT_VAR", int, 0,
        )

        # Whether to warn about kernel launches where the grid size will
        # under utilize the GPU due to low occupancy. On by default.
        CUDA_LOW_OCCUPANCY_WARNINGS = _readenv(
            "NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS", int, 1)

        # Whether to use the official CUDA Python API Bindings
        CUDA_USE_NVIDIA_BINDING = _readenv(
            "NUMBA_CUDA_USE_NVIDIA_BINDING", int, 0)

        # Debug flag to control compiler debug print
        DEBUG = _readenv("NUMBA_DEBUG", int, 0)

        # DEBUG print IR after pass names
        DEBUG_PRINT_AFTER = _readenv("NUMBA_DEBUG_PRINT_AFTER", str, "none")

        # DEBUG print IR before pass names
        DEBUG_PRINT_BEFORE = _readenv("NUMBA_DEBUG_PRINT_BEFORE", str, "none")

        # DEBUG print IR before and after pass names
        DEBUG_PRINT_WRAP = _readenv("NUMBA_DEBUG_PRINT_WRAP", str, "none")

        # Highlighting in intermediate dumps
        HIGHLIGHT_DUMPS = _readenv("NUMBA_HIGHLIGHT_DUMPS", int, 0)

        # JIT Debug flag to trigger IR instruction print
        DEBUG_JIT = _readenv("NUMBA_DEBUG_JIT", int, 0)

        # Enable debugging of front-end operation
        # (up to and including IR generation)
        DEBUG_FRONTEND = _readenv("NUMBA_DEBUG_FRONTEND", int, 0)

        # Enable debug prints in nrtdynmod
        DEBUG_NRT = _readenv("NUMBA_DEBUG_NRT", int, 0)

        # How many recently deserialized functions to retain regardless
        # of external references
        FUNCTION_CACHE_SIZE = _readenv("NUMBA_FUNCTION_CACHE_SIZE", int, 128)

        # Maximum tuple size that parfors will unpack and pass to
        # internal gufunc.
        PARFOR_MAX_TUPLE_SIZE = _readenv("NUMBA_PARFOR_MAX_TUPLE_SIZE",
                                         int, 100)

        # Enable logging of cache operation
        DEBUG_CACHE = _readenv("NUMBA_DEBUG_CACHE", int, DEBUG)

        # Redirect cache directory
        # Contains path to the directory
        CACHE_DIR = _readenv("NUMBA_CACHE_DIR", str, "")

        # Enable tracing support
        TRACE = _readenv("NUMBA_TRACE", int, 0)

        # Enable chrome tracing support
        CHROME_TRACE = _readenv("NUMBA_CHROME_TRACE", str, "")

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
                           DEBUG_FRONTEND)

        # Force dump of Numba IR in SSA form
        DUMP_SSA = _readenv("NUMBA_DUMP_SSA", int,
                            DEBUG_FRONTEND or DEBUG_TYPEINFER)

        # print debug info of analysis and optimization on array operations
        DEBUG_ARRAY_OPT = _readenv("NUMBA_DEBUG_ARRAY_OPT", int, 0)

        # insert debug stmts to print information at runtime
        DEBUG_ARRAY_OPT_RUNTIME = _readenv(
            "NUMBA_DEBUG_ARRAY_OPT_RUNTIME", int, 0)

        # print stats about parallel for-loops
        DEBUG_ARRAY_OPT_STATS = _readenv("NUMBA_DEBUG_ARRAY_OPT_STATS", int, 0)

        # prints user friendly information about parallel
        PARALLEL_DIAGNOSTICS = _readenv("NUMBA_PARALLEL_DIAGNOSTICS", int, 0)

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

        # Switch on  superword-level parallelism vectorization, default is on.
        SLP_VECTORIZE = _readenv("NUMBA_SLP_VECTORIZE", int, 1)

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

        # x86-64 specific
        # Enable AVX on supported platforms where it won't degrade performance.
        def avx_default():
            if not _os_supports_avx():
                return False
            else:
                # There are various performance issues with AVX and LLVM
                # on some CPUs (list at
                # http://llvm.org/bugs/buglist.cgi?quicksearch=avx).
                # For now we'd rather disable it, since it can pessimize code
                cpu_name = ll.get_host_cpu_name()
                return cpu_name not in ('corei7-avx', 'core-avx-i',
                                        'sandybridge', 'ivybridge')

        ENABLE_AVX = _readenv("NUMBA_ENABLE_AVX", int, avx_default)

        # if set and SVML is available, it will be disabled
        # By default, it's disabled on 32-bit platforms.
        DISABLE_INTEL_SVML = _readenv(
            "NUMBA_DISABLE_INTEL_SVML", int, IS_32BITS)

        # Disable jit for debugging
        DISABLE_JIT = _readenv("NUMBA_DISABLE_JIT", int, 0)

        # choose parallel backend to use
        THREADING_LAYER_PRIORITY = _readenv(
            "NUMBA_THREADING_LAYER_PRIORITY",
            lambda string: string.split(),
            ['tbb', 'omp', 'workqueue'],
        )
        THREADING_LAYER = _readenv("NUMBA_THREADING_LAYER", str, 'default')

        CAPTURED_ERRORS = _readenv("NUMBA_CAPTURED_ERRORS",
                                   _validate_captured_errors_style,
                                   'old_style')

        # CUDA Configs

        # Whether to warn about kernel launches where a host array
        # is used as a parameter, forcing a copy to and from the device.
        # On by default.
        CUDA_WARN_ON_IMPLICIT_COPY = _readenv(
            "NUMBA_CUDA_WARN_ON_IMPLICIT_COPY", int, 1)

        # Force CUDA compute capability to a specific version
        FORCE_CUDA_CC = _readenv("NUMBA_FORCE_CUDA_CC", _parse_cc, None)

        # The default compute capability to target when compiling to PTX.
        CUDA_DEFAULT_PTX_CC = _readenv("NUMBA_CUDA_DEFAULT_PTX_CC", _parse_cc,
                                       (5, 3))

        # Disable CUDA support
        DISABLE_CUDA = _readenv("NUMBA_DISABLE_CUDA",
                                int, int(MACHINE_BITS == 32))

        # Enable CUDA simulator
        ENABLE_CUDASIM = _readenv("NUMBA_ENABLE_CUDASIM", int, 0)

        # CUDA logging level
        # Any level name from the *logging* module.  Case insensitive.
        # Defaults to CRITICAL if not set or invalid.
        # Note: This setting only applies when logging is not configured.
        #       Any existing logging configuration is preserved.
        CUDA_LOG_LEVEL = _readenv("NUMBA_CUDA_LOG_LEVEL", str, '')

        # Include argument values in the CUDA Driver API logs
        CUDA_LOG_API_ARGS = _readenv("NUMBA_CUDA_LOG_API_ARGS", int, 0)

        # Maximum number of pending CUDA deallocations (default: 10)
        CUDA_DEALLOCS_COUNT = _readenv("NUMBA_CUDA_MAX_PENDING_DEALLOCS_COUNT",
                                       int, 10)

        # Maximum ratio of pending CUDA deallocations to capacity (default: 0.2)
        CUDA_DEALLOCS_RATIO = _readenv("NUMBA_CUDA_MAX_PENDING_DEALLOCS_RATIO",
                                       float, 0.2)

        CUDA_ARRAY_INTERFACE_SYNC = _readenv("NUMBA_CUDA_ARRAY_INTERFACE_SYNC",
                                             int, 1)

        # Path of the directory that the CUDA driver libraries are located
        CUDA_DRIVER = _readenv("NUMBA_CUDA_DRIVER", str, '')

        # Buffer size for logs produced by CUDA driver operations (e.g.
        # linking)
        CUDA_LOG_SIZE = _readenv("NUMBA_CUDA_LOG_SIZE", int, 1024)

        # Whether to generate verbose log messages when JIT linking
        CUDA_VERBOSE_JIT_LOG = _readenv("NUMBA_CUDA_VERBOSE_JIT_LOG", int, 1)

        # Whether the default stream is the per-thread default stream
        CUDA_PER_THREAD_DEFAULT_STREAM = _readenv(
            "NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM", int, 0)

        # Location of the CUDA include files
        if IS_WIN32:
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                default_cuda_include_path = os.path.join(cuda_path, "include")
            else:
                default_cuda_include_path = "cuda_include_not_found"
        else:
            default_cuda_include_path = os.path.join(os.sep, 'usr', 'local',
                                                     'cuda', 'include')
        CUDA_INCLUDE_PATH = _readenv("NUMBA_CUDA_INCLUDE_PATH", str,
                                     default_cuda_include_path)

        # Threading settings

        # The default number of threads to use.
        def num_threads_default():
            try:
                sched_getaffinity = os.sched_getaffinity
            except AttributeError:
                pass
            else:
                return max(1, len(sched_getaffinity(0)))

            cpu_count = os.cpu_count()
            if cpu_count is not None:
                return max(1, cpu_count)

            return 1

        NUMBA_DEFAULT_NUM_THREADS = num_threads_default()

        # Numba thread pool size (defaults to number of CPUs on the system).
        _NUMBA_NUM_THREADS = _readenv("NUMBA_NUM_THREADS", int,
                                      NUMBA_DEFAULT_NUM_THREADS)
        if ('NUMBA_NUM_THREADS' in globals()
                and globals()['NUMBA_NUM_THREADS'] != _NUMBA_NUM_THREADS):

            from numba.np.ufunc import parallel
            if parallel._is_initialized:
                raise RuntimeError("Cannot set NUMBA_NUM_THREADS to a "
                                   "different value once the threads have been "
                                   "launched (currently have %s, "
                                   "trying to set %s)" %
                                   (_NUMBA_NUM_THREADS,
                                    globals()['NUMBA_NUM_THREADS']))

        NUMBA_NUM_THREADS = _NUMBA_NUM_THREADS
        del _NUMBA_NUM_THREADS

        # Profiling support

        # Indicates if a profiler detected. Only VTune can be detected for now
        RUNNING_UNDER_PROFILER = 'VS_PROFILER' in os.environ

        # Enables jit events in LLVM to support profiling of dynamic code
        ENABLE_PROFILING = _readenv(
            "NUMBA_ENABLE_PROFILING", int, int(RUNNING_UNDER_PROFILER))

        # Debug Info

        # The default value for the `debug` flag
        DEBUGINFO_DEFAULT = _readenv("NUMBA_DEBUGINFO", int, ENABLE_PROFILING)
        CUDA_DEBUGINFO_DEFAULT = _readenv("NUMBA_CUDA_DEBUGINFO", int, 0)

        EXTEND_VARIABLE_LIFETIMES = _readenv("NUMBA_EXTEND_VARIABLE_LIFETIMES",
                                             int, 0)

        # gdb binary location
        def which_gdb(path_or_bin):
            gdb = shutil.which(path_or_bin)
            return gdb if gdb is not None else path_or_bin

        GDB_BINARY = _readenv("NUMBA_GDB_BINARY", which_gdb, 'gdb')

        # CUDA Memory management
        CUDA_MEMORY_MANAGER = _readenv("NUMBA_CUDA_MEMORY_MANAGER", str,
                                       'default')

        # Experimental refprune pass
        LLVM_REFPRUNE_PASS = _readenv(
            "NUMBA_LLVM_REFPRUNE_PASS", int, 1,
        )
        LLVM_REFPRUNE_FLAGS = _readenv(
            "NUMBA_LLVM_REFPRUNE_FLAGS", str,
            "all" if LLVM_REFPRUNE_PASS else "",
        )

        # Timing support.

        # LLVM_PASS_TIMINGS enables LLVM recording of pass timings.
        LLVM_PASS_TIMINGS = _readenv(
            "NUMBA_LLVM_PASS_TIMINGS", int, 0,
        )

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
