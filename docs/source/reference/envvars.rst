.. _numba-envvars:

Environment variables
=====================

Numba allows its behaviour to be changed through the use of environment
variables. Unless otherwise mentioned, those variables have integer values and
default to zero.

For convenience, Numba also supports the use of a configuration file to persist
configuration settings. Note: To use this feature ``pyyaml`` must be installed.

The configuration file must be named ``.numba_config.yaml`` and be present in
the directory from which the Python interpreter is invoked. The configuration
file, if present, is read for configuration settings before the environment
variables are searched. This means that the environment variable settings will
override the settings obtained from a configuration file (the configuration file
is for setting permanent preferences whereas the environment variables are for
ephemeral preferences).

The format of the configuration file is a dictionary in ``YAML`` format that
maps the environment variables below (without the ``NUMBA_`` prefix) to a
desired value. For example, to permanently switch on developer mode
(``NUMBA_DEVELOPER_MODE`` environment variable) and control flow graph printing
(``NUMBA_DUMP_CFG`` environment variable), create a configuration file with the
contents::

    developer_mode: 1
    dump_cfg: 1

This can be especially useful in the case of wanting to use a set color scheme
based on terminal background color. For example, if the terminal background
color is black, the ``dark_bg`` color scheme would be well suited and can be set
for permanent use by adding::

    color_scheme: dark_bg


Debugging
---------

These variables influence what is printed out during compilation of
:term:`JIT functions <JIT function>`.

.. envvar:: NUMBA_DEVELOPER_MODE

    If set to non-zero, developer mode produces full tracebacks and disables
    help instructions. Default is zero.

.. envvar:: NUMBA_FULL_TRACEBACKS

    If set to non-zero, enable full tracebacks when an exception occurs.
    Defaults to the value set by `NUMBA_DEVELOPER_MODE`.

.. envvar:: NUMBA_SHOW_HELP

    If not set or set to zero, show user level help information.
    Defaults to the negation of the value set by `NUMBA_DEVELOPER_MODE`.

.. envvar:: NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING

    If set to non-zero error message highlighting is disabled. This is useful
    for running the test suite on CI systems.

.. envvar:: NUMBA_COLOR_SCHEME

   Alters the color scheme used in error reporting (requires the ``colorama``
   package to be installed to work). Valid values are:

   - ``no_color`` No color added, just bold font weighting.
   - ``dark_bg`` Suitable for terminals with a dark background.
   - ``light_bg`` Suitable for terminals with a light background.
   - ``blue_bg`` Suitable for terminals with a blue background.
   - ``jupyter_nb`` Suitable for use in Jupyter Notebooks.

   *Default value:* ``no_color``. The type of the value is ``string``.

.. envvar:: NUMBA_DISABLE_PERFORMANCE_WARNINGS

   If set to non-zero the issuing of performance warnings is disabled. Default
   is zero.

.. envvar:: NUMBA_DEBUG

   If set to non-zero, print out all possible debugging information during
   function compilation.  Finer-grained control can be obtained using other
   variables below.

.. envvar:: NUMBA_DEBUG_FRONTEND

   If set to non-zero, print out debugging information during operation
   of the compiler frontend, up to and including generation of the Numba
   Intermediate Representation.

.. envvar:: NUMBA_DEBUGINFO

   If set to non-zero, enable debug for the full application by setting
   the default value of the ``debug`` option in ``jit``. Beware that
   enabling debug info significantly increases the memory consumption
   for each compiled function.
   Default value equals to the value of `NUMBA_ENABLE_PROFILING`.

.. envvar:: NUMBA_GDB_BINARY

   Set the ``gdb`` binary for use in Numba's ``gdb`` support, this takes the
   form  of a path and full name of the binary, for example:
   ``/path/from/root/to/binary/name_of_gdb_binary`` This is to permit
   the use of a ``gdb`` from a non-default location with a non-default name. If
   not set ``gdb`` is assumed to reside at ``/usr/bin/gdb``.

.. envvar:: NUMBA_DEBUG_TYPEINFER

   If set to non-zero, print out debugging information about type inference.

.. envvar:: NUMBA_ENABLE_PROFILING

   Enables JIT events of LLVM in order to support profiling of jitted functions.
   This option is automatically enabled under certain profilers.

.. envvar:: NUMBA_TRACE

   If set to non-zero, trace certain function calls (function entry and exit
   events, including arguments and return values).

.. envvar:: NUMBA_DUMP_BYTECODE

   If set to non-zero, print out the Python :py:term:`bytecode` of
   compiled functions.

.. envvar:: NUMBA_DUMP_CFG

   If set to non-zero, print out information about the Control Flow Graph
   of compiled functions.

.. envvar:: NUMBA_DUMP_IR

   If set to non-zero, print out the Numba Intermediate Representation
   of compiled functions.

.. envvar:: NUMBA_DEBUG_PRINT_AFTER

   Dump the Numba IR after declared pass(es). This is useful for debugging IR
   changes made by given passes. Accepted values are:

   * Any pass name (as given by the ``.name()`` method on the class)
   * Multiple pass names as a comma separated list, i.e. ``"foo_pass,bar_pass"``
   * The token ``"all"``, which will print after all passes.

   The default value is ``"none"`` so as to prevent output.

.. envvar:: NUMBA_DUMP_ANNOTATION

   If set to non-zero, print out types annotations for compiled functions.

.. envvar:: NUMBA_DUMP_LLVM

   Dump the unoptimized LLVM assembly source of compiled functions.
   Unoptimized code is usually very verbose; therefore,
   :envvar:`NUMBA_DUMP_OPTIMIZED` is recommended instead.

.. envvar:: NUMBA_DUMP_FUNC_OPT

   Dump the LLVM assembly source after the LLVM "function optimization"
   pass, but before the "module optimization" pass.  This is useful mostly
   when developing Numba itself, otherwise use :envvar:`NUMBA_DUMP_OPTIMIZED`.

.. envvar:: NUMBA_DUMP_OPTIMIZED

   Dump the LLVM assembly source of compiled functions after all
   optimization passes.  The output includes the raw function as well as
   its CPython-compatible wrapper (whose name begins with ``wrapper.``).
   Note that the function is often inlined inside the wrapper, as well.

.. envvar:: NUMBA_DEBUG_ARRAY_OPT

   Dump debugging information related to the processing associated with
   the ``parallel=True`` jit decorator option.

.. envvar:: NUMBA_DEBUG_ARRAY_OPT_RUNTIME

   Dump debugging information related to the runtime scheduler associated
   with the ``parallel=True`` jit decorator option.

.. envvar:: NUMBA_DEBUG_ARRAY_OPT_STATS

   Dump statistics about how many operators/calls are converted to
   parallel for-loops and how many are fused together, which are associated
   with the ``parallel=True`` jit decorator option.

.. envvar:: NUMBA_PARALLEL_DIAGNOSTICS

   If set to an integer value between 1 and 4 (inclusive) diagnostic information
   about parallel transforms undertaken by Numba will be written to STDOUT. The
   higher the value set the more detailed the information produced.

.. envvar:: NUMBA_DUMP_ASSEMBLY

   Dump the native assembly code of compiled functions.

.. seealso::
   :ref:`numba-troubleshooting` and :ref:`architecture`.


Compilation options
-------------------

.. envvar:: NUMBA_OPT

   The optimization level; this option is passed straight to LLVM.

   *Default value:* 3

.. envvar:: NUMBA_LOOP_VECTORIZE

   If set to non-zero, enable LLVM loop vectorization.

   *Default value:* 1 (except on 32-bit Windows)

.. envvar:: NUMBA_ENABLE_AVX

   If set to non-zero, enable AVX optimizations in LLVM.  This is disabled
   by default on Sandy Bridge and Ivy Bridge architectures as it can sometimes
   result in slower code on those platforms.

.. envvar:: NUMBA_DISABLE_INTEL_SVML

    If set to non-zero and Intel SVML is available, the use of SVML will be
    disabled.

.. envvar:: NUMBA_COMPATIBILITY_MODE

   If set to non-zero, compilation of JIT functions will never entirely
   fail, but instead generate a fallback that simply interprets the
   function.  This is only to be used if you are migrating a large
   codebase from an old Numba version (before 0.12), and want to avoid
   breaking everything at once.  Otherwise, please don't use this.

.. envvar:: NUMBA_DISABLE_JIT

   Disable JIT compilation entirely.  The :func:`~numba.jit` decorator acts
   as if it performs no operation, and the invocation of decorated functions
   calls the original Python function instead of a compiled version.  This
   can be useful if you want to run the Python debugger over your code.

.. envvar:: NUMBA_CPU_NAME
.. envvar:: NUMBA_CPU_FEATURES

    Override CPU and CPU features detection.
    By setting ``NUMBA_CPU_NAME=generic``, a generic CPU model is picked
    for the CPU architecture and the feature list (``NUMBA_CPU_FEATURES``)
    defaults to empty.  CPU features must be listed with the format
    ``+feature1,-feature2`` where ``+`` indicates enable and ``-`` indicates
    disable. For example, ``+sse,+sse2,-avx,-avx2`` enables SSE and SSE2, and
    disables AVX and AVX2.

    These settings are passed to LLVM for configuring the compilation target.
    To get a list of available options, use the ``llc`` commandline tool
    from LLVM, for example::

        llc -march=x86 -mattr=help


    .. tip:: To force all caching functions (``@jit(cache=True)``) to emit
        portable code (portable within the same architecture and OS),
        simply set ``NUMBA_CPU_NAME=generic``.

.. envvar:: NUMBA_FUNCTION_CACHE_SIZE

    Override the size of the function cache for retaining recently
    deserialized functions in memory.  In systems like
    `Dask <http://dask.pydata.org>`_, it is common for functions to be deserialized
    multiple times.  Numba will cache functions as long as there is a
    reference somewhere in the interpreter.  This cache size variable controls
    how many functions that are no longer referenced will also be retained,
    just in case they show up in the future.  The implementation of this is
    not a true LRU, but the large size of the cache should be sufficient for
    most situations.

    Note: this is unrelated to the compilation cache.

    *Default value:* 128


.. _numba-envvars-caching:

Caching options
---------------

Options for the compilation cache.

.. envvar:: NUMBA_DEBUG_CACHE

   If set to non-zero, print out information about operation of the
   :ref:`JIT compilation cache <jit-cache>`.

.. envvar:: NUMBA_CACHE_DIR

    Override the location of the cache directory. If defined, this should be
    a valid directory path.

    If not defined, Numba picks the cache directory in the following order:

    1. In-tree cache. Put the cache next to the corresponding source file under
       a ``__pycache__`` directory following how ``.pyc`` files are stored.
    2. User-wide cache. Put the cache in the user's application directory using
       ``appdirs.user_cache_dir`` from the
       `Appdirs package <https://github.com/ActiveState/appdirs>`_.
    3. IPython cache. Put the cache in an IPython specific application
       directory.
       Stores are made under the ``numba_cache`` in the directory returned by
       ``IPython.paths.get_ipython_cache_dir()``.

    Also see :ref:`docs on cache sharing <cache-sharing>` and
    :ref:`docs on cache clearing <cache-clearing>`



GPU support
-----------

.. envvar:: NUMBA_DISABLE_CUDA

   If set to non-zero, disable CUDA support.

.. envvar:: NUMBA_FORCE_CUDA_CC

   If set, force the CUDA compute capability to the given version (a
   string of the type ``major.minor``), regardless of attached devices.

.. envvar:: NUMBA_ENABLE_CUDASIM

   If set, don't compile and execute code for the GPU, but use the CUDA
   Simulator instead. For debugging purposes.

Threading Control
-----------------

.. envvar:: NUMBA_NUM_THREADS

   If set, the number of threads in the thread pool for the parallel CPU target
   will take this value. Must be greater than zero. This value is independent
   of ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``.

   *Default value:* The number of CPU cores on the system as determined at run
   time, this can be accessed via ``numba.config.NUMBA_DEFAULT_NUM_THREADS``.

.. envvar:: NUMBA_THREADING_LAYER

   This environment variable controls the library used for concurrent execution
   for the CPU parallel targets (``@vectorize(target='parallel')``,
   ``@guvectorize(target='parallel')``  and ``@njit(parallel=True)``). The
   variable type is string and by default is ``default`` which will select a
   threading layer based on what is available in the runtime. The valid values
   are (for more information about these see
   :ref:`the threading layer documentation <numba-threading-layer>`):

   * ``default`` - select a threading layer based on what is available in the
     current runtime.
   * ``safe`` - select a threading layer that is both fork and thread safe
     (requires the TBB package).
   * ``forksafe`` - select a threading layer that is fork safe.
   * ``threadsafe`` - select a threading layer that is thread safe.
   * ``tbb`` - A threading layer backed by Intel TBB.
   * ``omp`` - A threading layer backed by OpenMP.
   * ``workqueue`` - A simple built-in work-sharing task scheduler.

