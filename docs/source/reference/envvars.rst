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


Errors and warnings display
---------------------------

.. envvar:: NUMBA_WARNINGS

   If set to non-zero, printout of Numba warnings is enabled, otherwise
   the warnings are suppressed.  The warnings can give insight into the
   compilation process.


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

.. envvar:: NUMBA_DEBUG

   If set to non-zero, print out all possible debugging information during
   function compilation.  Finer-grained control can be obtained using other
   variables below.

.. envvar:: NUMBA_DEBUG_FRONTEND

   If set to non-zero, print out debugging information during operation
   of the compiler frontend, up to and including generation of the Numba
   Intermediate Representation.

.. envvar:: NUMBA_DEBUG_TYPEINFER

   If set to non-zero, print out debugging information about type inference.

.. envvar:: NUMBA_DEBUG_CACHE

   If set to non-zero, print out information about operation of the
   :ref:`JIT compilation cache <jit-cache>`.

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

.. envvar:: NUMBA_DUMP_ANNOTATION

   If set to non-zero, print out types annotations for compiled functions.

.. envvar:: NUMBA_DUMP_LLVM

   Dump the unoptimized LLVM assembler source of compiled functions.
   Unoptimized code is usually very verbose; therefore,
   :envvar:`NUMBA_DUMP_OPTIMIZED` is recommended instead.

.. envvar:: NUMBA_DUMP_FUNC_OPT

   Dump the LLVM assembler source after the LLVM "function optimization"
   pass, but before the "module optimization" pass.  This is useful mostly
   when developing Numba itself, otherwise use :envvar:`NUMBA_DUMP_OPTIMIZED`.

.. envvar:: NUMBA_DUMP_OPTIMIZED

   Dump the LLVM assembler source of compiled functions after all
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

.. envvar:: NUMBA_DUMP_ASSEMBLY

   Dump the native assembler code of compiled functions.

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

.. envvar:: NUMBA_CPU_NAME and NUMBA_CPU_FEATURES

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
