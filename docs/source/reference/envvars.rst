.. _numba-envvars:

Environment variables
=====================

.. note:: This section relates to environment variables that impact Numba's
          runtime, for compile time environment variables see
          :ref:`numba-source-install-env_vars`.

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

Jit flags
---------

These variables globally override flags to the :func:`~numba.jit` decorator.

.. envvar:: NUMBA_BOUNDSCHECK

   If set to 0 or 1, globally disable or enable bounds checking, respectively.
   The default if the variable is not set or set to an empty string is to use
   the ``boundscheck`` flag passed to the :func:`~numba.jit` decorator for a
   given function. See the documentation of :ref:`@jit
   <jit-decorator-boundscheck>` for more information.

   Note, due to limitations in numba, the bounds checking currently produces
   exception messages that do not match those from NumPy. If you set
   ``NUMBA_FULL_TRACEBACKS=1``, the full exception message with the axis,
   index, and shape information will be printed to the terminal.

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

    If set to non-zero, show resources for getting help. Default is zero.

.. envvar:: NUMBA_CAPTURED_ERRORS

    Alters the way in which Numba captures and handles exceptions that do not
    inherit from ``numba.core.errors.NumbaError`` during compilation (e.g.
    standard Python exceptions). This does not impact runtime exception
    handling. Valid values are:

    - ``"old_style"`` (default): this is the exception handling behaviour that
      is present in Numba versions <= 0.54.x. Numba will capture and wrap all
      errors occuring in compilation and depending on the compilation phase they
      will likely materialize as part of the message in a ``TypingError`` or a
      ``LoweringError``.
    - ``"new_style"`` this will treat any exception that does not inherit from
      ``numba.core.errors.NumbaError`` **and** is raised during compilation as a
      "hard error", i.e. the exception will propagate and compilation will halt.
      The purpose of this new style is to differentiate between intentionally
      raised exceptions and those which occur due to mistakes. For example, if
      an ``AttributeError`` occurs in the typing of an ``@overload`` function,
      under this new behaviour it is assumed that this a mistake in the
      implementation and compilation will halt due to this exception. This
      behaviour will eventually become the default.

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

.. envvar:: NUMBA_HIGHLIGHT_DUMPS

   If set to non-zero and ``pygments`` is installed, syntax highlighting is
   applied to Numba IR, LLVM IR and assembly dumps. Default is zero.

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

.. envvar:: NUMBA_EXTEND_VARIABLE_LIFETIMES

    If set to non-zero, extend the lifetime of variables to the end of the block
    in which their lifetime ends. This is particularly useful in conjunction
    with `NUMBA_DEBUGINFO` as it helps with introspection of values. Default is
    zero.

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


.. envvar:: NUMBA_DUMP_SSA

   If set to non-zero, print out the Numba Intermediate Representation of
   compiled functions after conversion to Static Single Assignment (SSA) form.

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

.. envvar:: NUMBA_LLVM_PASS_TIMINGS

    Set to ``1`` to enable recording of pass timings in LLVM;
    e.g. ``NUMBA_LLVM_PASS_TIMINGS=1``.
    See :ref:`developer-llvm-timings`.

    *Default value*: ``0`` (Off)

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

.. envvar:: NUMBA_SLP_VECTORIZE

   If set to non-zero, enable LLVM superword-level parallelism vectorization.

   *Default value:* 1

.. envvar:: NUMBA_ENABLE_AVX

   If set to non-zero, enable AVX optimizations in LLVM.  This is disabled
   by default on Sandy Bridge and Ivy Bridge architectures as it can sometimes
   result in slower code on those platforms.

.. envvar:: NUMBA_DISABLE_INTEL_SVML

    If set to non-zero and Intel SVML is available, the use of SVML will be
    disabled.

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

.. envvar:: NUMBA_LLVM_REFPRUNE_PASS

    Turns on the LLVM pass level reference-count pruning pass and disables the
    regex based implementation in Numba.

    *Default value:* 1 (On)

.. envvar:: NUMBA_LLVM_REFPRUNE_FLAGS

    When ``NUMBA_LLVM_REFPRUNE_PASS`` is on, this allows configuration
    of subpasses in the reference-count pruning LLVM pass.

    Valid values are any combinations of the below separated by `,`
    (case-insensitive):

    - ``all``: enable all subpasses.
    - ``per_bb``: enable per-basic-block level pruning, which is same as the
      old regex based implementation.
    - ``diamond``: enable inter-basic-block pruning that is a diamond shape
      pattern, i.e. a single-entry single-exit CFG subgraph where has an incref
      in the entry and a corresponding decref in the exit.
    - ``fanout``: enable inter-basic-block pruning that has a fanout pattern,
      i.e. a single-entry multiple-exit CFG subgraph where the entry has an
      incref and every exit has a corresponding decref.
    - ``fanout_raise``: same as ``fanout`` but allow subgraph exit nodes to be
      raising an exception and not have a corresponding decref.

    For example, ``all`` is the same as
    ``per_bb, diamond, fanout, fanout_raise``

    *Default value:* "all"


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


.. _numba-envvars-gpu-support:

GPU support
-----------

.. envvar:: NUMBA_DISABLE_CUDA

   If set to non-zero, disable CUDA support.

.. envvar:: NUMBA_FORCE_CUDA_CC

   If set, force the CUDA compute capability to the given version (a
   string of the type ``major.minor``), regardless of attached devices.

.. envvar:: NUMBA_CUDA_DEFAULT_PTX_CC

   The default compute capability (a string of the type ``major.minor``) to
   target when compiling to PTX using ``cuda.compile_ptx``. The default is
   5.2, which is the lowest non-deprecated compute capability in the most
   recent version of the CUDA toolkit supported (10.2 at present).

.. envvar:: NUMBA_ENABLE_CUDASIM

   If set, don't compile and execute code for the GPU, but use the CUDA
   Simulator instead. For debugging purposes.


.. envvar:: NUMBA_CUDA_ARRAY_INTERFACE_SYNC

   Whether to synchronize on streams provided by objects imported using the CUDA
   Array Interface. This defaults to 1. If set to 0, then no synchronization
   takes place, and the user of Numba (and other CUDA libraries) is responsible
   for ensuring correctness with respect to synchronization on streams.

.. envvar:: NUMBA_CUDA_LOG_LEVEL

   For debugging purposes. If no other logging is configured, the value of this
   variable is the logging level for CUDA API calls. The default value is
   ``CRITICAL`` - to trace all API calls on standard error, set this to
   ``DEBUG``.

.. envvar:: NUMBA_CUDA_LOG_API_ARGS

   By default the CUDA API call logs only give the names of functions called.
   Setting this variable to 1 also includes the values of arguments to Driver
   API calls in the logs.

.. envvar:: NUMBA_CUDA_DRIVER

   Path of the directory in which the CUDA driver libraries are to be found.
   Normally this should not need to be set as Numba can locate the driver in
   standard locations. However, this variable can be used if the driver is in a
   non-standard location.

.. envvar:: NUMBA_CUDA_LOG_SIZE

   Buffer size for logs produced by CUDA driver API operations. This defaults
   to 1024 and should not normally need to be modified - however, if an error
   in an API call produces a large amount of output that appears to be
   truncated (perhaps due to multiple long function names, for example) then
   this variable can be used to increase the buffer size and view the full
   error message.

.. envvar:: NUMBA_CUDA_VERBOSE_JIT_LOG

   Whether the CUDA driver should produce verbose log messages. Defaults to 1,
   indicating that verbose messaging is enabled. This should not need to be
   modified under normal circumstances.

.. envvar:: NUMBA_CUDA_PER_THREAD_DEFAULT_STREAM

   When set to 1, the default stream is the per-thread default stream. When set
   to 0, the default stream is the legacy default stream. This defaults to 0,
   for the legacy default stream. It may default to 1 in a future release of
   Numba. See `Stream Synchronization Behavior
   <https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html>`_
   for an explanation of the legacy and per-thread default streams.

.. envvar:: NUMBA_NPY_RELAXED_STRIDES_CHECKING

   By default arrays that inherit from ``numba.misc.dummyarray.Array`` (e.g.
   CUDA device arrays) compute their contiguity using relaxed strides checking,
   which is the default mechanism used by NumPy since version 1.12
   (see `NPY_RELAXED_STRIDES_CHECKING
   <https://numpy.org/doc/stable/release/1.8.0-notes.html#npy-relaxed-strides-checking>`_).
   Setting ``NUMBA_NPY_RELAXED_STRIDES_CHECKING=0`` reverts back to strict
   strides checking. This option should not normally be needed, but is provided
   in case it is needed to work around latent bugs related to strict strides
   checking.

   Strict strides checking is deprecated and may be removed in future. See
   :ref:`deprecation-strict-strides`.

.. envvar:: NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS

   Enable warnings if the grid size is too small relative to the number of
   streaming multiprocessors (SM). This option is on by default (default value is 1).

   The heuristic checked is whether ``gridsize < 2 * (number of SMs)``. NOTE: The absence of
   a warning does not imply a good gridsize relative to the number of SMs. Disabling
   this warning will reduce the number of CUDA API calls (during JIT compilation), as the
   heuristic needs to check the number of SMs available on the device in the
   current context.

.. envvar:: CUDA_WARN_ON_IMPLICIT_COPY

   Enable warnings if a kernel is launched with host memory which forces a copy to and
   from the device. This option is on by default (default value is 1).


Threading Control
-----------------

.. envvar:: NUMBA_NUM_THREADS

   If set, the number of threads in the thread pool for the parallel CPU target
   will take this value. Must be greater than zero. This value is independent
   of ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS``.

   *Default value:* The number of CPU cores on the system as determined at run
   time. This can be accessed via :obj:`numba.config.NUMBA_DEFAULT_NUM_THREADS`.

   See also the section on :ref:`setting_the_number_of_threads` for
   information on how to set the number of threads at runtime.

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

.. envvar:: NUMBA_THREADING_LAYER_PRIORITY

   This environment variable controls the order in which the libraries used for
   concurrent execution, for the CPU parallel targets
   (``@vectorize(target='parallel')``, ``@guvectorize(target='parallel')``
   and ``@njit(parallel=True)``), are prioritized for use. The variable type is
   string and by default is ``tbb omp workqueue``, with the priority taken based
   on position from the left of the string, left most being the highest. Valid
   values are any permutation of the three choices (for more information about
   these see :ref:`the threading layer documentation <numba-threading-layer>`.)
