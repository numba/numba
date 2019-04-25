A Map of the Numba Repository
=============================

The Numba repository is quite large, and due to age has functionality spread
around many locations.  To help orient developers, this document will try to
summarize where different categories of functionality can be found.

.. note::
    It is likely that the organization of the code base will change in the
    future to improve organization.  Follow `issue #3807 <https://github.com/numba/numba/issues/3807>`_
    for more details.


Support Files
-------------

Build and Packaging
'''''''''''''''''''

- ``setup.py`` - Standard Python distutils/setuptools script
- ``MANIFEST.in`` - Distutils packaging instructions
- ``requirements.txt`` - Pip package requirements, not used by conda
- ``versioneer.py`` - Handles automatic setting of version in
  installed package from git tags
- ``.flake8`` - Preferences for code formatting.  Files should be
  fixed and removed from the exception list as time allows.
- ``buildscripts/condarecipe.local`` - Conda build recipe
- ``buildscripts/remove_unwanted_files.py`` - Helper script to remove
  files that will not compile under Python 2. Used by build recipes.
- ``buildscripts/condarecipe_clone_icc_rt`` - Recipe to build a
  standalone icc_rt package.


Continuous Integration
''''''''''''''''''''''
- ``.binstar.yml`` - Binstar Build CI config (inactive)
- ``azure-pipelines.yml`` - Azure Pipelines CI config (active:
  Win/Mac/Linux)
- ``buildscripts/azure/`` - Azure Pipeline configuration for specific
  platforms
- ``.travis.yml`` - Travis CI config (active: Mac/Linux, will be
  dropped in the future)
- ``appveyor.yml`` - Appveyor CI config (inactive: Win)
- ``buildscripts/appveyor/`` - Appveyor build scripts
- ``buildscripts/incremental/`` - Generic scripts for building Numba
  on various CI systems
- ``codecov.yml`` - Codecov.io coverage reporting


Documentation / Examples
''''''''''''''''''''''''
- ``LICENSE`` - License for Numba
- ``LICENSES.third-party`` - License for third party code vendored
  into Numba
- ``README.rst`` - README for repo, also uploaded to PyPI
- ``CONTRIBUTING.md`` - Documentation on how to contribute to project
  (out of date, should be updated to point to Sphinx docs)
- ``AUTHORS`` - List of Github users who have contributed PRs (out of
  date)
- ``CHANGE_LOG`` - History of Numba releases, also directly embedded
  into Sphinx documentation
- ``docs/`` - Documentation source
- ``docs/_templates/`` - Directory for templates (to override defaults
  with Sphinx theme)
- ``docs/Makefile`` - Used to build Sphinx docs with ``make``
- ``docs/source`` - ReST source for Numba documentation
- ``docs/_static/`` - Static CSS and image assets for Numba docs
- ``docs/gh-pages.py`` - Utility script to update Numba docs (stored
  as gh-pages)
- ``docs/make.bat`` - Not used (remove?)
- ``examples/`` - Example scripts demonstrating numba (re/move to
  numba-examples repo?)
- ``examples/notebooks/`` - Example notebooks (re/move to
  numba-examples repo?)
- ``benchmarks/`` - Benchmark scripts (re/move to numba-examples
  repo?)
- ``tutorials/`` - Tutorial notebooks (definitely out of date, should
  remove and direct to external tutorials)
- ``numba/scripts/generate_lower_listing.py`` - Dump all registered
  implementations decorated with ``@lower*`` for reference
  documentation.  Currently misses implementations from the higher
  level extension API.



Numba Source Code
-----------------

Numba ships with both the source code and tests in one package.

- ``numba/`` - all of the source code and tests


Public API
''''''''''

These define aspects of the public Numba interface.

- ``numba/decorators.py`` - User-facing decorators for compiling
  regular functions on the CPU
- ``numba/extending.py`` - Public decorators for extending Numba
  (``overload``, ``intrinsic``, etc)
- ``numba/ccallback.py`` - ``@cfunc`` decorator for compiling
  functions to a fixed C signature.  Used to make callbacks.
- ``numba/npyufunc/decorators.py`` - ufunc/gufunc compilation
  decorators
- ``numba/config.py`` - Numba global config options and environment
  variable handling
- ``numba/annotations`` - Gathering and printing type annotations of
  Numba IR
- ``numba/pretty_annotate.py`` - Code highlighting of Numba functions
  and types (both ANSI terminal and HTML)


Dispatching
'''''''''''

- ``numba/dispatcher.py`` - Dispatcher objects are compiled functions
  produced by ``@jit``.  A dispatcher has different implementations
  for different type signatures.
- ``numba/_dispatcher.{h,c}`` - C interface to C++ dispatcher
  implementation
- ``numba/_dispatcherimpl.cpp`` - C++ dispatcher implementation (for
  speed on common data types)


Compiler Pipeline
'''''''''''''''''

- ``numba/compiler.py`` - Compiler pipelines and flags
- ``numba/errors.py`` - Numba exception and warning classes 
- ``numba/ir.py`` - Numba IR data structure objects
- ``numba/bytecode.py`` - Bytecode parsing and function identity (??)
- ``numba/interpreter.py`` - Translate Python interpreter bytecode to
  Numba IR
- ``numba/analysis.py`` - Utility functions to analyze Numba IR
  (variable lifetime, prune branches, etc)
- ``numba/dataflow.py`` - Dataflow analysis for Python bytecode (used
  in analysis.py)
- ``numba/controlflow.py`` - Control flow analysis of Numba IR and
  Python bytecode
- ``numba/typeinfer.py`` - Type inference algorithm
- ``numba/transforms.py`` - Numba IR transformations
- ``numba/rewrites`` - Rewrite passes used by compiler
- ``numba/rewrites/__init__.py`` - Loads all rewrite passes so they
  are put into the registry
- ``numba/rewrites/registry.py`` - Registry object for collecting
  rewrite passes
- ``numba/rewrites/ir_print.py`` - Write print() calls into special
  print nodes in the IR
- ``numba/rewrites/static_raise.py`` - Converts exceptions with static
  arguments into a special form that can be lowered
- ``numba/rewrites/macros.py`` - Generic support for macro expansion
  in the Numba IR
- ``numba/rewrites/static_getitem.py`` - Rewrites getitem and setitem
  with constant arguments to allow type inference
- ``numba/rewrites/static_binop.py`` - Rewrites binary operations
  (specifically ``**``) with constant arguments so faster code can be
  generated
- ``numba/inline_closurecall.py`` - Inlines body of closure functions
  to call site.  Support for array comprehensions, reduction inlining, 
  and stencil inlining.
- ``numba/macro.py`` - Alias to ``numba.rewrites.macros``
- ``numba/postproc.py`` - Postprocessor for Numba IR that computes
  variable lifetime, inserts del operations, and handles generators
- ``numba/lowering.py`` - General implementation of lowering Numba IR
  to LLVM
- ``numba/withcontexts.py`` - General scaffolding for implementing
  context managers in nopython mode, and the objectmode context
  manager
- ``numba/pylowering.py`` - Lowering of Numba IR in object mode
- ``numba/pythonapi.py`` - LLVM IR code generation to interface with
  CPython API


Type Management
'''''''''''''''

- ``numba/typeconv/`` - Implementation of type casting and type
  signature matching in both C++ and Python
- ``numba/capsulethunk.h`` - Used by typeconv
- ``numba/types/`` - definition of the Numba type hierarchy, used
  everywhere in compiler to select implementations
- ``numba/consts.py`` - Constant inference (used to make constant
  values available during codegen when possible)
- ``numba/datamodel`` - LLVM IR representations of data types in
  different contexts
- ``numba/datamodel/models.py`` - Models for most standard types
- ``numba/datamodel/registry.py`` - Decorator to register new data
  models
- ``numba/datamodel/packer.py`` - Pack typed values into a data
  structure
- ``numba/datamodel/testing.py`` - Data model tests (this should
  move??)
- ``numba/datamodel/manager.py`` - Map types to data models


Compiled Extensions
'''''''''''''''''''

Numba uses a small amount of compiled C/C++ code for core
functionality, like dispatching and type matching where performance
matters, and it is more convenient to encapsulate direct interaction
with CPython APIs.

- ``numba/_arraystruct.h`` - Struct for holding NumPy array
  attributes.  Used in helperlib and the Numba Runtime.
- ``numba/_helperlib.c`` - C functions required by Numba compiled code
  at runtime.  Linked into ahead-of-time compiled modules
- ``numba/_helpermod.c`` - Python extension module with pointers to
  functions from ``_helperlib.c`` and ``_npymath_exports.c``
- ``numba/_npymath_exports.c`` - Export function pointer table to
  NumPy C math functions
- ``numba/_dynfuncmod.c`` - Python extension module exporting
  _dynfunc.c functionality
- ``numba/_dynfunc.c`` - C level Environment and Closure objects (keep
  in sync with numba/target/base.py)
- ``numba/mathnames.h`` - Macros for defining names of math functions
- ``numba/_pymodule.h`` - C macros for Python 2/3 portable naming of C
  API functions
- ``numba/_math_c99.{h,c}`` - C99 math compatibility (needed Python
  2.7 on Windows, compiled with VS2008)
- ``numba/mviewbuf.c`` - Handles Python memoryviews
- ``numba/_typeof.{h,c}`` - C implementation of type fingerprinting,
  used by dispatcher
- ``numba/_numba_common.h`` - Portable C macro for marking symbols
  that can be shared between object files, but not outside the
  library.



Misc Support
''''''''''''

- ``numba/_version.py`` - Updated by versioneer
- ``numba/runtime`` - Language runtime.  Currently manages
  reference-counted memory allocated on the heap by Numba-compiled
  functions
- ``numba/ir_utils.py`` - Utility functions for working with Numba IR
  data structures
- ``numba/cgutils.py`` - Utility functions for generating common code
  patterns in LLVM IR
- ``numba/six.py`` - Vendored subset of ``six`` package for Python 2 +
  3 compatibility
- ``numba/io_support.py`` - Workaround for various names of StringIO
  in different Python versions (should this be in six?)
- ``numba/utils.py`` - Python 2 backports of Python 3 functionality
  (also imports local copy of ``six``)
- ``numba/appdirs.py`` - Vendored package for determining application
  config directories on every platform
- ``numba/compiler_lock.py`` - Global compiler lock because Numba's usage
  of LLVM is not thread-safe
- ``numba/special.py`` - Python stub implementations of special Numba
  functions (prange, gdb*)
- ``numba/servicelib/threadlocal.py`` - Thread-local stack used by GPU
  targets
- ``numba/servicelib/service.py`` - Should be removed?
- ``numba/itanium_mangler.py`` - Python implementation of Itanium C++
  name mangling
- ``numba/findlib.py`` - Helper function for locating shared libraries
  on all platforms
- ``numba/debuginfo.py`` - Helper functions to construct LLVM IR debug
  info
- ``numba/unsafe`` - ``@intrinsic`` helper functions that can be used
  to implement direct memory/pointer manipulation from nopython mode
  functions
- ``numba/unsafe/refcount.py`` - Read reference count of object
- ``numba/unsafe/tuple.py`` - Replace a value in a tuple slot
- ``numba/unsafe/ndarray.py`` - NumPy array helpers
- ``numba/unsafe/bytes.py`` - Copying and dereferencing data from void
  pointers
- ``numba/dummyarray.py`` - Used by GPU backends to hold array information
  on the host, but not the data.
- ``numba/callwrapper.py`` - Handles argument unboxing and releasing
  the GIL when moving from Python to nopython mode
- ``numba/ctypes_support.py`` - Import this instead of ``ctypes`` to
  workaround portability issue with Python 2.7
- ``numba/cffi_support.py`` - Alias of numba.typing.cffi_utils for
  backward compatibility (still needed?)
- ``numba/numpy_support.py`` - Helper functions for working with NumPy
  and translating Numba types to and from NumPy dtypes.
- ``numba/tracing.py`` - Decorator for tracing Python calls and
  emitting log messages
- ``numba/funcdesc.py`` - Classes for describing function metadata
  (used in the compiler)
- ``numba/sigutils.py`` - Helper functions for parsing and normalizing
  Numba type signatures
- ``numba/serialize.py`` - Support for pickling compiled functions
- ``numba/caching.py`` - Disk cache for compiled functions
- ``numba/npdatetime.py`` - Helper functions for implementing NumPy
  datetime64 support


Core Python Data Types
''''''''''''''''''''''

- ``numba/_hashtable.{h,c}`` - Adaptation of the Python 3.7 hash table
  implementation
- ``numba/_dictobject.{h,c}`` - C level implementation of typed
  dictionary
- ``numba/dictobject.py`` - Nopython mode wrapper for typed dictionary
- ``numba/unicode.py`` - Unicode strings (Python 3.5 and later)
- ``numba/typed`` - Python interfaces to statically typed containers
- ``numba/typed/typeddict.py`` - Python interface to typed dictionary
- ``numba/jitclass`` - Implementation of JIT compilation of Python
  classes
- ``numba/generators.py`` - Support for lowering Python generators


Math
''''

- ``numba/_random.c`` - Reimplementation of NumPy / CPython random
  number generator
- ``numba/_lapack.c`` - Wrappers for calling BLAS and LAPACK functions
  (requires SciPy)


ParallelAccelerator
'''''''''''''''''''

Code transformation passes that extract parallelizable code from
a function and convert it into multithreaded gufunc calls.

- ``numba/parfor.py`` - General ParallelAccelerator
- ``numba/stencil.py`` - Stencil function decorator (implemented
  without ParallelAccelerator)
- ``numba/stencilparfor.py`` - ParallelAccelerator implementation of
  stencil
- ``numba/array_analysis.py`` - Array analysis passes used in
  ParallelAccelerator


Deprecated Functionality
''''''''''''''''''''''''

- ``numba/smartarray.py`` - Experiment with an array object that has
  both CPU and GPU backing.  Should be removed in future.


Debugging Support
'''''''''''''''''

- ``numba/targets/gdb_hook.py`` - Hooks to jump into GDB from nopython
  mode
- ``numba/targets/cmdlang.gdb`` - Commands to setup GDB for setting
  explicit breakpoints from Python


Type Signatures (CPU)
'''''''''''''''''''''

Some (usually older) Numba supported functionality separates the
declaration of allowed type signatures from the definition of
implementations.  This package contains registries of type signatures
that must be matched during type inference.

- ``numba/typing`` - Type signature module
- ``numba/typing/templates.py`` - Base classes for type signature
  templates
- ``numba/typing/cmathdecl.py`` - Python complex math (``cmath``)
  module
- ``numba/typing/bufproto.py`` - Interpreting objects supporting the
  buffer protocol
- ``numba/typing/mathdecl.py`` - Python ``math`` module
- ``numba/typing/listdecl.py`` - Python lists
- ``numba/typing/builtins.py`` - Python builtin global functions and
  operators
- ``numba/typing/randomdecl.py`` - Python and NumPy ``random`` modules
- ``numba/typing/setdecl.py`` - Python sets
- ``numba/typing/npydecl.py`` - NumPy ndarray (and operators), NumPy
  functions
- ``numba/typing/arraydecl.py`` - Python ``array`` module
- ``numba/typing/context.py`` - Implementation of typing context
  (class that collects methods used in type inference)
- ``numba/typing/collections.py`` - Generic container operations and
  namedtuples
- ``numba/typing/ctypes_utils.py`` - Typing ctypes-wrapped function
  pointers
- ``numba/typing/enumdecl.py`` - Enum types
- ``numba/typing/cffi_utils.py`` - Typing of CFFI objects
- ``numba/typing/typeof.py`` - Implementation of typeof operations
  (maps Python object to Numba type)
- ``numba/typing/npdatetime.py`` - Datetime dtype support for NumPy
  arrays


Target Implementations (CPU)
''''''''''''''''''''''''''''

Implementations of Python / NumPy functions and some data models.
These modules are responsible for generating LLVM IR during lowering.
Note that some of these modules do not have counterparts in the typing
package because newer Numba extension APIs (like overload) allow
typing and implementation to be specified together.

- ``numba/targets`` - Implementations of compilable operations
- ``numba/targets/cpu.py`` - Context for code gen on CPU
- ``numba/targets/base.py`` - Base class for all target contexts
- ``numba/targets/codegen.py`` - Driver for code generation
- ``numba/targets/boxing.py`` - Boxing and unboxing for most data
  types
- ``numba/targets/intrinsics.py`` - Utilities for converting LLVM
  intrinsics to other math calls
- ``numba/targets/callconv.py`` - Implements different calling
  conventions for Numba-compiled functions
- ``numba/targets/iterators.py`` - Iterable data types and iterators
- ``numba/targets/hashing.py`` - Hashing algorithms
- ``numba/targets/ufunc_db.py`` - Big table mapping types to ufunc
  implementations
- ``numba/targets/setobj.py`` - Python set type
- ``numba/targets/options.py`` - Container for options that control
  lowering
- ``numba/targets/printimpl.py`` - Print function
- ``numba/targets/smartarray.py`` - Smart array (deprecated)
- ``numba/targets/cmathimpl.py`` - Python complex math module
- ``numba/targets/optional.py`` - Special type representing value or
  ``None``
- ``numba/targets/tupleobj.py`` - Tuples (statically typed as
  immutable struct)
- ``numba/targets/mathimpl.py`` - Python ``math`` module
- ``numba/targets/heapq.py`` - Python ``heapq`` module
- ``numba/targets/registry.py`` - Registry object for collecting
  implementations for a specific target
- ``numba/targets/imputils.py`` - Helper functions for lowering
- ``numba/targets/builtins.py`` - Python builtin functions and
  operators
- ``numba/targets/externals.py`` - Registers external C functions
  needed to link generated code
- ``numba/targets/quicksort.py`` - Quicksort implementation used with
  list and array objects
- ``numba/targets/mergesort.py`` - Mergesort implementation used with
  array objects
- ``numba/targets/randomimpl.py`` - Python and NumPy ``random``
  modules
- ``numba/targets/npyimpl.py`` - Implementations of most NumPy ufuncs
- ``numba/targets/slicing.py`` - Slice objects, and index calculations
  used in slicing
- ``numba/targets/numbers.py`` - Numeric values (int, float, etc)
- ``numba/targets/listobj.py`` - Python lists
- ``numba/targets/fastmathpass.py`` - Rewrite pass to add fastmath
  attributes to function call sites and binary operations
- ``numba/targets/removerefctpass.py`` - Rewrite pass to remove
  unnecessary incref/decref pairs
- ``numba/targets/cffiimpl.py`` - CFFI functions
- ``numba/targets/descriptors.py`` - empty base class for all target
  descriptors (is this needed?)
- ``numba/targets/arraymath.py`` - Math operations on arrays (both
  Python and NumPy)
- ``numba/targets/linalg.py`` - NumPy linear algebra operations
- ``numba/targets/rangeobj.py`` - Python `range` objects
- ``numba/targets/npyfuncs.py`` - Kernels used in generating some
  NumPy ufuncs
- ``numba/targets/arrayobj.py`` - Array operations (both NumPy and
  buffer protocol)
- ``numba/targets/enumimpl.py`` - Enum objects
- ``numba/targets/polynomial.py`` - ``numpy.roots`` function
- ``numba/targets/npdatetime.py`` - NumPy datetime operations


Ufunc Compiler and Runtime
''''''''''''''''''''''''''

- ``numba/npyufunc`` - ufunc compiler implementation
- ``numba/npyufunc/_internal.{h,c}`` - Python extension module with
  helper functions that use CPython & NumPy C API
- ``numba/npyufunc/_ufunc.c`` - Used by `_internal.c`
- ``numba/npyufunc/deviceufunc.py`` - Custom ufunc dispatch for
  non-CPU targets
- ``numba/npyufunc/gufunc_scheduler.{h,cpp}`` - Schedule work chunks
  to threads
- ``numba/npyufunc/dufunc.py`` - Special ufunc that can compile new
  implementations at call time
- ``numba/npyufunc/ufuncbuilder.py`` - Top-level orchestration of
  ufunc/gufunc compiler pipeline
- ``numba/npyufunc/sigparse.py`` - Parser for generalized ufunc
  indexing signatures
- ``numba/npyufunc/parfor.py`` - gufunc lowering for
  ParallelAccelerator
- ``numba/npyufunc/parallel.py`` - Codegen for ``parallel`` target
- ``numba/npyufunc/array_exprs.py`` - Rewrite pass for turning array
  expressions in regular functions into ufuncs
- ``numba/npyufunc/wrappers.py`` - Wrap scalar function kernel with
  loops
- ``numba/npyufunc/workqueue.{h,c}`` - Threading backend based on
  pthreads/Windows threads and queues
- ``numba/npyufunc/omppool.cpp`` - Threading backend based on OpenMP
- ``numba/npyufunc/tbbpool.cpp`` - Threading backend based on TBB



Unit Tests (CPU)
''''''''''''''''

CPU unit tests (GPU target unit tests listed in later sections

- ``runtests.py`` - Convenience script that launches test runner and
  turns on full compiler tracebacks
- ``run_coverage.py`` - Runs test suite with coverage tracking enabled
- ``.coveragerc`` - Coverage.py configuration
- ``numba/runtests.py`` - Entry point to unittest runner
- ``numba/_runtests.py`` - Implementation of custom test runner
  command line interface
- ``numba/tests/test_*`` - Test cases
- ``numba/tests/*_usecases.py`` - Python functions compiled by some
  unit tests
- ``numba/tests/support.py`` - Helper functions for testing and
  special TestCase implementation
- ``numba/tests/dummy_module.py`` - Module used in
  ``test_dispatcher.py``
- ``numba/tests/npyufunc`` - ufunc / gufunc compiler tests
- ``numba/unittest_support.py`` - Import instead of unittest to handle
  portability issues
- ``numba/testing`` - Support code for testing
- ``numba/testing/ddt.py`` - decorators for test cases
- ``numba/testing/loader.py`` - Find tests on disk
- ``numba/testing/notebook.py`` - Support for testing notebooks
- ``numba/testing/main.py`` - Numba test runner


Command Line Utilities
''''''''''''''''''''''
- ``bin/numba`` - Command line stub, delegates to main in
  ``numba_entry.py``
- ``numba/numba_entry.py`` - Main function for ``numba`` command line
  tool
- ``numba/pycc`` - Ahead of time compilation of functions to shared
  library extension
- ``numba/pycc/__init__.py`` - Main function for ``pycc`` command line
  tool
- ``numba/pycc/cc.py`` - User-facing API for tagging functions to
  compile ahead of time
- ``numba/pycc/compiler.py`` - Compiler pipeline for creating
  standalone Python extension modules
- ``numba/pycc/llvm_types.py`` - Aliases to LLVM data types used by
  ``compiler.py``
- ``numba/pycc/pycc`` - Stub to call main function.  Is this still
  used?
- ``numba/pycc/modulemixin.c`` - C file compiled into every compiled
  extension.  Pulls in C source from Numba core that is needed to make
  extension standalone
- ``numba/pycc/platform.py`` - Portable interface to platform-specific
  compiler toolchains
- ``numba/pycc/decorators.py`` - Deprecated decorators for tagging
  functions to compile.  Use ``cc.py`` instead.


CUDA GPU Target
'''''''''''''''

Note that the CUDA target does reuse some parts of the CPU target.

- ``numba/cuda/`` - The implementation of the CUDA (NVIDIA GPU) target
  and associated unit tests
- ``numba/cuda/decorators.py`` - Compiler decorators for CUDA kernels
  and device functions
- ``numba/cuda/dispatcher.py`` - Dispatcher for CUDA JIT functions
- ``numba/cuda/printimpl.py`` - Special implementation of device printing
- ``numba/cuda/libdevice.py`` - Registers libdevice functions
- ``numba/cuda/kernels/`` - Custom kernels for reduction and transpose 
- ``numba/cuda/device_init.py`` - Initializes the CUDA target when
  imported
- ``numba/cuda/compiler.py`` - Compiler pipeline for CUDA target
- ``numba/cuda/intrinsic_wrapper.py`` - CUDA device intrinsics
  (shuffle, ballot, etc)
- ``numba/cuda/initialize.py`` - Defered initialization of the CUDA
  device and subsystem.  Called only when user imports ``numba.cuda``
- ``numba/cuda/simulator_init.py`` - Initalizes the CUDA simulator
  subsystem (only when user requests it with env var)
- ``numba/cuda/random.py`` - Implementation of random number generator
- ``numba/cuda/api.py`` - User facing APIs imported into ``numba.cuda.*``
- ``numba/cuda/stubs.py`` - Python placeholders for functions that
  only can be used in GPU device code
- ``numba/cuda/simulator/`` - Simulate execution of CUDA kernels in
  Python interpreter
- ``numba/cuda/vectorizers.py`` - Subclasses of ufunc/gufunc compilers
  for CUDA
- ``numba/cuda/args.py`` - Management of kernel arguments, including
  host<->device transfers
- ``numba/cuda/target.py`` - Typing and target contexts for GPU
- ``numba/cuda/cudamath.py`` - Type signatures for math functions in
  CUDA Python
- ``numba/cuda/errors.py`` - Validation of kernel launch configuration
- ``numba/cuda/nvvmutils.py`` - Helper functions for generating
  NVVM-specific IR
- ``numba/cuda/testing.py`` - Support code for creating CUDA unit
  tests and capturing standard out
- ``numba/cuda/cudadecl.py`` - Type signatures of CUDA API (threadIdx,
  blockIdx, atomics) in Python on GPU
- ``numba/cuda/cudaimpl.py`` - Implementations of CUDA API functions
  on GPU
- ``numba/cuda/codegen.py`` - Code generator object for CUDA target
- ``numba/cuda/cudadrv/`` - Wrapper around CUDA driver API
- ``numba/cuda/tests/`` - CUDA unit tests, skipped when CUDA is not
  detected
- ``numba/cuda/tests/cudasim/`` - Tests of CUDA simulator
- ``numba/cuda/tests/nocuda/`` - Tests for NVVM functionality when
  CUDA not present
- ``numba/cuda/tests/cudapy/`` - Tests of compiling Python functions
  for GPU
- ``numba/cuda/tests/cudadrv/`` - Tests of Python wrapper around CUDA
  API


ROCm GPU Target
'''''''''''''''

Note that the ROCm target does reuse some parts of the CPU target, and
duplicates some code from CUDA target.  A future refactoring could
pull out the common subset of CUDA and ROCm.  An older version of this
target was based on the HSA API, so "hsa" appears in many places.

- ``numba/roc`` - ROCm GPU target for AMD GPUs
- ``numba/roc/descriptor.py`` - TargetDescriptor subclass for ROCm
  target
- ``numba/roc/enums.py`` - Internal constants
- ``numba/roc/mathdecl.py`` - Declarations of math functions that can
  be used on device
- ``numba/roc/mathimpl.py`` - Implementations of math functions for
  device
- ``numba/roc/compiler.py`` - Compiler pipeline for ROCm target
- ``numba/roc/hlc`` - Wrapper around LLVM interface for AMD GPU
- ``numba/roc/initialize.py`` - Register ROCm target for ufunc/gufunc
  compiler
- ``numba/roc/hsadecl.py`` - Type signatures for ROCm device API in
  Python
- ``numba/roc/hsaimpl.py`` - Implementations of ROCm device API
- ``numba/roc/dispatch.py`` - ufunc/gufunc dispatcher
- ``numba/roc/README.md`` - Notes on testing target (should be
  deleted)
- ``numba/roc/api.py`` - Host API for ROCm actions
- ``numba/roc/gcn_occupancy.py`` - Heuristic to compute occupancy of
  kernels
- ``numba/roc/stubs.py`` - Host stubs for device functions
- ``numba/roc/vectorizers.py`` - Builds ufuncs
- ``numba/roc/target.py`` - Target and typing contexts
- ``numba/roc/hsadrv`` - Python wrapper around ROCm (based on HSA)
  driver API calls
- ``numba/roc/codegen.py`` - Codegen subclass for ROCm target
- ``numba/roc/decorators.py`` - ``@jit`` decorator for kernels and
  device functions
- ``numba/roc/tests`` - Unit tests for ROCm target
- ``numba/roc/tests/hsapy`` - Tests of compiling ROCm kernels written
  in Python syntax
- ``numba/roc/tests/hsadrv`` - Tests of Python wrapper on platform
  API.
