A Map of the Numba Repository
=============================

The Numba repository is quite large, and due to age has functionality spread
around many locations.  To help orient developers, this document will try to
summarize where different categories of functionality can be found.


Support Files
-------------

Build and Packaging
'''''''''''''''''''

- :ghfile:`setup.py` - Standard Python distutils/setuptools script
- :ghfile:`MANIFEST.in` - Distutils packaging instructions
- :ghfile:`requirements.txt` - Pip package requirements, not used by conda
- :ghfile:`versioneer.py` - Handles automatic setting of version in
  installed package from git tags
- :ghfile:`.flake8` - Preferences for code formatting.  Files should be
  fixed and removed from the exception list as time allows.
- :ghfile:`.pre-commit-config.yaml` - Configuration file for pre-commit hooks.
- :ghfile:`.readthedocs.yml` - Configuration file for Read the Docs.
- :ghfile:`buildscripts/condarecipe.local` - Conda build recipe


Continuous Integration
''''''''''''''''''''''
- :ghfile:`azure-pipelines.yml` - Azure Pipelines CI config (active:
  Win/Mac/Linux)
- :ghfile:`buildscripts/azure/` - Azure Pipeline configuration for specific
  platforms
- :ghfile:`buildscripts/incremental/` - Generic scripts for building Numba
  on various CI systems
- :ghfile:`codecov.yml` - Codecov.io coverage reporting


Documentation
'''''''''''''
- :ghfile:`LICENSE` - License for Numba
- :ghfile:`LICENSES.third-party` - License for third party code vendored
  into Numba
- :ghfile:`README.rst` - README for repo, also uploaded to PyPI
- :ghfile:`CONTRIBUTING.md` - Documentation on how to contribute to project
  (out of date, should be updated to point to Sphinx docs)
- :ghfile:`CHANGE_LOG` - History of Numba releases, also directly embedded
  into Sphinx documentation
- :ghfile:`docs/` - Documentation source
- :ghfile:`docs/_templates/` - Directory for templates (to override defaults
  with Sphinx theme)
- :ghfile:`docs/Makefile` - Used to build Sphinx docs with ``make``
- :ghfile:`docs/source` - ReST source for Numba documentation
- :ghfile:`docs/_static/` - Static CSS and image assets for Numba docs
- :ghfile:`docs/make.bat` - Not used (remove?)
- :ghfile:`docs/requirements.txt` - Pip package requirements for building docs
  with Read the Docs.
- :ghfile:`numba/scripts/generate_lower_listing.py` - Dump all registered
  implementations decorated with ``@lower*`` for reference
  documentation.  Currently misses implementations from the higher
  level extension API.


Numba Source Code
-----------------

Numba ships with both the source code and tests in one package.

- :ghfile:`numba/` - all of the source code and tests


Public API
''''''''''

These define aspects of the public Numba interface.

- :ghfile:`numba/core/decorators.py` - User-facing decorators for compiling
  regular functions on the CPU
- :ghfile:`numba/core/extending.py` - Public decorators for extending Numba
  (``overload``, ``intrinsic``, etc)
  - :ghfile:`numba/experimental/structref.py` - Public API for defining a mutable struct
- :ghfile:`numba/core/ccallback.py` - ``@cfunc`` decorator for compiling
  functions to a fixed C signature.  Used to make callbacks.
- :ghfile:`numba/np/ufunc/decorators.py` - ufunc/gufunc compilation
  decorators
- :ghfile:`numba/core/config.py` - Numba global config options and environment
  variable handling
- :ghfile:`numba/core/annotations` - Gathering and printing type annotations of
  Numba IR
- :ghfile:`numba/core/annotations/pretty_annotate.py` - Code highlighting of
  Numba functions and types (both ANSI terminal and HTML)
- :ghfile:`numba/core/event.py` - A simple event system for applications to
  listen to specific compiler events.


Dispatching
'''''''''''

- :ghfile:`numba/core/dispatcher.py` - Dispatcher objects are compiled functions
  produced by ``@jit``.  A dispatcher has different implementations
  for different type signatures.
- :ghfile:`numba/_dispatcher.cpp` - C++ dispatcher implementation (for speed on
  common data types)
- :ghfile:`numba/core/retarget.py` - Support for dispatcher objects to switch
  target via a specific with-context.


Compiler Pipeline
'''''''''''''''''

- :ghfile:`numba/core/compiler.py` - Compiler pipelines and flags
- :ghfile:`numba/core/errors.py` - Numba exception and warning classes
- :ghfile:`numba/core/ir.py` - Numba IR data structure objects
- :ghfile:`numba/core/bytecode.py` - Bytecode parsing and function identity (??)
- :ghfile:`numba/core/interpreter.py` - Translate Python interpreter bytecode to
  Numba IR
- :ghfile:`numba/core/analysis.py` - Utility functions to analyze Numba IR
  (variable lifetime, prune branches, etc)
- :ghfile:`numba/core/controlflow.py` - Control flow analysis of Numba IR and
  Python bytecode
- :ghfile:`numba/core/typeinfer.py` - Type inference algorithm
- :ghfile:`numba/core/transforms.py` - Numba IR transformations
- :ghfile:`numba/core/rewrites` - Rewrite passes used by compiler
- :ghfile:`numba/core/rewrites/__init__.py` - Loads all rewrite passes so they
  are put into the registry
- :ghfile:`numba/core/rewrites/registry.py` - Registry object for collecting
  rewrite passes
- :ghfile:`numba/core/rewrites/ir_print.py` - Write print() calls into special
  print nodes in the IR
- :ghfile:`numba/core/rewrites/static_raise.py` - Converts exceptions with
  static arguments into a special form that can be lowered
- :ghfile:`numba/core/rewrites/static_getitem.py` - Rewrites getitem and setitem
  with constant arguments to allow type inference
- :ghfile:`numba/core/rewrites/static_binop.py` - Rewrites binary operations
  (specifically ``**``) with constant arguments so faster code can be
  generated
- :ghfile:`numba/core/inline_closurecall.py` - Inlines body of closure functions
  to call site.  Support for array comprehensions, reduction inlining,
  and stencil inlining.
- :ghfile:`numba/core/postproc.py` - Postprocessor for Numba IR that computes
  variable lifetime, inserts del operations, and handles generators
- :ghfile:`numba/core/lowering.py` - General implementation of lowering Numba IR
  to LLVM
  :ghfile:`numba/core/environment.py` - Runtime environment object
- :ghfile:`numba/core/withcontexts.py` - General scaffolding for implementing
  context managers in nopython mode, and the objectmode context
  manager
- :ghfile:`numba/core/pylowering.py` - Lowering of Numba IR in object mode
- :ghfile:`numba/core/pythonapi.py` - LLVM IR code generation to interface with
  CPython API
- :ghfile:`numba/core/targetconfig.py` - Utils for target configurations such
  as compiler flags.


Type Management
'''''''''''''''

- :ghfile:`numba/core/typeconv/` - Implementation of type casting and type
  signature matching in both C++ and Python
- :ghfile:`numba/capsulethunk.h` - Used by typeconv
- :ghfile:`numba/core/types/` - definition of the Numba type hierarchy, used
  everywhere in compiler to select implementations
- :ghfile:`numba/core/consts.py` - Constant inference (used to make constant
  values available during codegen when possible)
- :ghfile:`numba/core/datamodel` - LLVM IR representations of data types in
  different contexts
- :ghfile:`numba/core/datamodel/models.py` - Models for most standard types
- :ghfile:`numba/core/datamodel/registry.py` - Decorator to register new data
  models
- :ghfile:`numba/core/datamodel/packer.py` - Pack typed values into a data
  structure
- :ghfile:`numba/core/datamodel/testing.py` - Data model tests (this should
  move??)
- :ghfile:`numba/core/datamodel/manager.py` - Map types to data models


Compiled Extensions
'''''''''''''''''''

Numba uses a small amount of compiled C/C++ code for core
functionality, like dispatching and type matching where performance
matters, and it is more convenient to encapsulate direct interaction
with CPython APIs.

- :ghfile:`numba/_arraystruct.h` - Struct for holding NumPy array
  attributes.  Used in helperlib and the Numba Runtime.
- :ghfile:`numba/_helperlib.c` - C functions required by Numba compiled code
  at runtime.  Linked into ahead-of-time compiled modules
- :ghfile:`numba/_helpermod.c` - Python extension module with pointers to
  functions from ``_helperlib.c``
- :ghfile:`numba/_dynfuncmod.c` - Python extension module exporting
  _dynfunc.c functionality
- :ghfile:`numba/_dynfunc.c` - C level Environment and Closure objects (keep
  in sync with numba/target/base.py)
- :ghfile:`numba/mathnames.h` - Macros for defining names of math functions
- :ghfile:`numba/_pymodule.h` - C macros for Python 2/3 portable naming of C
  API functions
- :ghfile:`numba/mviewbuf.c` - Handles Python memoryviews
- :ghfile:`numba/_typeof.{h,cpp}` - C++ implementation of type fingerprinting,
  used by dispatcher
- :ghfile:`numba/_numba_common.h` - Portable C macro for marking symbols
  that can be shared between object files, but not outside the
  library.



Misc Support
''''''''''''

- :ghfile:`numba/_version.py` - Updated by versioneer
- :ghfile:`numba/core/runtime` - Language runtime.  Currently manages
  reference-counted memory allocated on the heap by Numba-compiled
  functions
- :ghfile:`numba/core/ir_utils.py` - Utility functions for working with Numba IR
  data structures
- :ghfile:`numba/core/cgutils.py` - Utility functions for generating common code
  patterns in LLVM IR
- :ghfile:`numba/core/utils.py` - Python 2 backports of Python 3 functionality
  (also imports local copy of ``six``)
- :ghfile:`numba/misc/appdirs.py` - Vendored package for determining application
  config directories on every platform
- :ghfile:`numba/core/compiler_lock.py` - Global compiler lock because Numba's
  usage of LLVM is not thread-safe
- :ghfile:`numba/misc/special.py` - Python stub implementations of special Numba
  functions (prange, gdb*)
- :ghfile:`numba/core/itanium_mangler.py` - Python implementation of Itanium C++
  name mangling
- :ghfile:`numba/misc/findlib.py` - Helper function for locating shared
  libraries on all platforms
- :ghfile:`numba/core/debuginfo.py` - Helper functions to construct LLVM IR
  debug
  info
- :ghfile:`numba/core/unsafe/refcount.py` - Read reference count of object
- :ghfile:`numba/core/unsafe/eh.py` - Exception handling helpers
- :ghfile:`numba/core/unsafe/nrt.py` - Numba runtime (NRT) helpers
- :ghfile:`numba/cpython/unsafe/tuple.py` - Replace a value in a tuple slot
- :ghfile:`numba/np/unsafe/ndarray.py` - NumPy array helpers
- :ghfile:`numba/core/unsafe/bytes.py` - Copying and dereferencing data from
  void pointers
- :ghfile:`numba/misc/dummyarray.py` - Used by GPU backends to hold array
  information on the host, but not the data.
- :ghfile:`numba/core/callwrapper.py` - Handles argument unboxing and releasing
  the GIL when moving from Python to nopython mode
- :ghfile:`numba/np/numpy_support.py` - Helper functions for working with NumPy
  and translating Numba types to and from NumPy dtypes.
- :ghfile:`numba/core/tracing.py` - Decorator for tracing Python calls and
  emitting log messages
- :ghfile:`numba/core/funcdesc.py` - Classes for describing function metadata
  (used in the compiler)
- :ghfile:`numba/core/sigutils.py` - Helper functions for parsing and
  normalizing Numba type signatures
- :ghfile:`numba/core/serialize.py` - Support for pickling compiled functions
- :ghfile:`numba/core/caching.py` - Disk cache for compiled functions
- :ghfile:`numba/np/npdatetime.py` - Helper functions for implementing NumPy
  datetime64 support
- :ghfile:`numba/misc/llvm_pass_timings.py` - Helper to record timings of
  LLVM passes.
- :ghfile:`numba/cloudpickle` - Vendored cloudpickle subpackage

Core Python Data Types
''''''''''''''''''''''

- :ghfile:`numba/_hashtable.{h,cpp}` - Adaptation of the Python 3.7 hash table
  implementation
- :ghfile:`numba/cext/dictobject.{h,c}` - C level implementation of typed
  dictionary
- :ghfile:`numba/typed/dictobject.py` - Nopython mode wrapper for typed
  dictionary
- :ghfile:`numba/cext/listobject.{h,c}` - C level implementation of typed list
- :ghfile:`numba/typed/listobject.py` - Nopython mode wrapper for typed list
- :ghfile:`numba/typed/typedobjectutils.py` - Common utilities for typed
  dictionary and list
- :ghfile:`numba/cpython/unicode.py` - Unicode strings (Python 3.5 and later)
- :ghfile:`numba/typed` - Python interfaces to statically typed containers
- :ghfile:`numba/typed/typeddict.py` - Python interface to typed dictionary
- :ghfile:`numba/typed/typedlist.py` - Python interface to typed list
- :ghfile:`numba/experimental/jitclass` - Implementation of experimental JIT
  compilation of Python classes
- :ghfile:`numba/core/generators.py` - Support for lowering Python generators


Math
''''

- :ghfile:`numba/_random.c` - Reimplementation of NumPy / CPython random
  number generator
- :ghfile:`numba/_lapack.c` - Wrappers for calling BLAS and LAPACK functions
  (requires SciPy)


ParallelAccelerator
'''''''''''''''''''

Code transformation passes that extract parallelizable code from
a function and convert it into multithreaded gufunc calls.

- :ghfile:`numba/parfors/parfor.py` - General ParallelAccelerator
- :ghfile:`numba/parfors/parfor_lowering.py` - gufunc lowering for
  ParallelAccelerator
- :ghfile:`numba/parfors/array_analysis.py` - Array analysis passes used in
  ParallelAccelerator


Stencil
'''''''

Implementation of ``@stencil``:

- :ghfile:`numba/stencils/stencil.py` - Stencil function decorator (implemented
  without ParallelAccelerator)
- :ghfile:`numba/stencils/stencilparfor.py` - ParallelAccelerator implementation
  of stencil


Debugging Support
'''''''''''''''''

- :ghfile:`numba/misc/gdb_hook.py` - Hooks to jump into GDB from nopython
  mode
- :ghfile:`numba/misc/cmdlang.gdb` - Commands to setup GDB for setting
  explicit breakpoints from Python


Type Signatures (CPU)
'''''''''''''''''''''

Some (usually older) Numba supported functionality separates the
declaration of allowed type signatures from the definition of
implementations.  This package contains registries of type signatures
that must be matched during type inference.

- :ghfile:`numba/core/typing` - Type signature module
- :ghfile:`numba/core/typing/templates.py` - Base classes for type signature
  templates
- :ghfile:`numba/core/typing/cmathdecl.py` - Python complex math (``cmath``)
  module
- :ghfile:`numba/core/typing/bufproto.py` - Interpreting objects supporting the
  buffer protocol
- :ghfile:`numba/core/typing/mathdecl.py` - Python ``math`` module
- :ghfile:`numba/core/typing/listdecl.py` - Python lists
- :ghfile:`numba/core/typing/builtins.py` - Python builtin global functions and
  operators
- :ghfile:`numba/core/typing/setdecl.py` - Python sets
- :ghfile:`numba/core/typing/npydecl.py` - NumPy ndarray (and operators), NumPy
  functions
- :ghfile:`numba/core/typing/arraydecl.py` - Python ``array`` module
- :ghfile:`numba/core/typing/context.py` - Implementation of typing context
  (class that collects methods used in type inference)
- :ghfile:`numba/core/typing/collections.py` - Generic container operations and
  namedtuples
- :ghfile:`numba/core/typing/ctypes_utils.py` - Typing ctypes-wrapped function
  pointers
- :ghfile:`numba/core/typing/enumdecl.py` - Enum types
- :ghfile:`numba/core/typing/cffi_utils.py` - Typing of CFFI objects
- :ghfile:`numba/core/typing/typeof.py` - Implementation of typeof operations
  (maps Python object to Numba type)
- :ghfile:`numba/core/typing/asnumbatype.py` - Implementation of
  ``as_numba_type`` operations (maps Python types to Numba type)
- :ghfile:`numba/core/typing/npdatetime.py` - Datetime dtype support for NumPy
  arrays


Target Implementations (CPU)
''''''''''''''''''''''''''''

Implementations of Python / NumPy functions and some data models.
These modules are responsible for generating LLVM IR during lowering.
Note that some of these modules do not have counterparts in the typing
package because newer Numba extension APIs (like overload) allow
typing and implementation to be specified together.

- :ghfile:`numba/core/cpu.py` - Context for code gen on CPU
- :ghfile:`numba/core/base.py` - Base class for all target contexts
- :ghfile:`numba/core/codegen.py` - Driver for code generation
- :ghfile:`numba/core/boxing.py` - Boxing and unboxing for most data
  types
- :ghfile:`numba/core/intrinsics.py` - Utilities for converting LLVM
  intrinsics to other math calls
- :ghfile:`numba/core/callconv.py` - Implements different calling
  conventions for Numba-compiled functions
- :ghfile:`numba/core/options.py` - Container for options that control
  lowering
- :ghfile:`numba/core/optional.py` - Special type representing value or
  ``None``
- :ghfile:`numba/core/registry.py` - Registry object for collecting
  implementations for a specific target
- :ghfile:`numba/core/imputils.py` - Helper functions for lowering
- :ghfile:`numba/core/externals.py` - Registers external C functions
  needed to link generated code
- :ghfile:`numba/core/fastmathpass.py` - Rewrite pass to add fastmath
  attributes to function call sites and binary operations
- :ghfile:`numba/core/removerefctpass.py` - Rewrite pass to remove
  unnecessary incref/decref pairs
- :ghfile:`numba/core/descriptors.py` - empty base class for all target
  descriptors (is this needed?)
- :ghfile:`numba/cpython/builtins.py` - Python builtin functions and
  operators
- :ghfile:`numba/cpython/cmathimpl.py` - Python complex math module
- :ghfile:`numba/cpython/enumimpl.py` - Enum objects
- :ghfile:`numba/cpython/hashing.py` - Hashing algorithms
- :ghfile:`numba/cpython/heapq.py` - Python ``heapq`` module
- :ghfile:`numba/cpython/iterators.py` - Iterable data types and iterators
- :ghfile:`numba/cpython/listobj.py` - Python lists
- :ghfile:`numba/cpython/mathimpl.py` - Python ``math`` module
- :ghfile:`numba/cpython/numbers.py` - Numeric values (int, float, etc)
- :ghfile:`numba/cpython/printimpl.py` - Print function
- :ghfile:`numba/cpython/randomimpl.py` - Python and NumPy ``random``
  modules
- :ghfile:`numba/cpython/rangeobj.py` - Python `range` objects
- :ghfile:`numba/cpython/slicing.py` - Slice objects, and index calculations
  used in slicing
- :ghfile:`numba/cpython/setobj.py` - Python set type
- :ghfile:`numba/cpython/tupleobj.py` - Tuples (statically typed as
  immutable struct)
- :ghfile:`numba/misc/cffiimpl.py` - CFFI functions
- :ghfile:`numba/misc/quicksort.py` - Quicksort implementation used with
  list and array objects
- :ghfile:`numba/misc/mergesort.py` - Mergesort implementation used with
  array objects
- :ghfile:`numba/np/arraymath.py` - Math operations on arrays (both
  Python and NumPy)
- :ghfile:`numba/np/arrayobj.py` - Array operations (both NumPy and
  buffer protocol)
- :ghfile:`numba/np/linalg.py` - NumPy linear algebra operations
- :ghfile:`numba/np/npdatetime.py` - NumPy datetime operations
- :ghfile:`numba/np/npyfuncs.py` - Kernels used in generating some
  NumPy ufuncs
- :ghfile:`numba/np/npyimpl.py` - Implementations of most NumPy ufuncs
- :ghfile:`numba/np/polynomial.py` - ``numpy.roots`` function
- :ghfile:`numba/np/ufunc_db.py` - Big table mapping types to ufunc
  implementations


Ufunc Compiler and Runtime
''''''''''''''''''''''''''

- :ghfile:`numba/np/ufunc` - ufunc compiler implementation
- :ghfile:`numba/np/ufunc/_internal.{h,c}` - Python extension module with
  helper functions that use CPython & NumPy C API
- :ghfile:`numba/np/ufunc/_ufunc.c` - Used by `_internal.c`
- :ghfile:`numba/np/ufunc/deviceufunc.py` - Custom ufunc dispatch for
  non-CPU targets
- :ghfile:`numba/np/ufunc/gufunc_scheduler.{h,cpp}` - Schedule work chunks
  to threads
- :ghfile:`numba/np/ufunc/dufunc.py` - Special ufunc that can compile new
  implementations at call time
- :ghfile:`numba/np/ufunc/ufuncbuilder.py` - Top-level orchestration of
  ufunc/gufunc compiler pipeline
- :ghfile:`numba/np/ufunc/sigparse.py` - Parser for generalized ufunc
  indexing signatures
- :ghfile:`numba/np/ufunc/parallel.py` - Codegen for ``parallel`` target
- :ghfile:`numba/np/ufunc/array_exprs.py` - Rewrite pass for turning array
  expressions in regular functions into ufuncs
- :ghfile:`numba/np/ufunc/wrappers.py` - Wrap scalar function kernel with
  loops
- :ghfile:`numba/np/ufunc/workqueue.{h,c}` - Threading backend based on
  pthreads/Windows threads and queues
- :ghfile:`numba/np/ufunc/omppool.cpp` - Threading backend based on OpenMP
- :ghfile:`numba/np/ufunc/tbbpool.cpp` - Threading backend based on TBB



Unit Tests (CPU)
''''''''''''''''

CPU unit tests (GPU target unit tests listed in later sections

- :ghfile:`runtests.py` - Convenience script that launches test runner and
  turns on full compiler tracebacks
- :ghfile:`.coveragerc` - Coverage.py configuration
- :ghfile:`numba/runtests.py` - Entry point to unittest runner
- :ghfile:`numba/testing/_runtests.py` - Implementation of custom test runner
  command line interface
- :ghfile:`numba/tests/test_*` - Test cases
- :ghfile:`numba/tests/*_usecases.py` - Python functions compiled by some
  unit tests
- :ghfile:`numba/tests/support.py` - Helper functions for testing and
  special TestCase implementation
- :ghfile:`numba/tests/dummy_module.py` - Module used in
  ``test_dispatcher.py``
- :ghfile:`numba/tests/npyufunc` - ufunc / gufunc compiler tests
- :ghfile:`numba/testing` - Support code for testing
- :ghfile:`numba/testing/loader.py` - Find tests on disk
- :ghfile:`numba/testing/notebook.py` - Support for testing notebooks
- :ghfile:`numba/testing/main.py` - Numba test runner


Command Line Utilities
''''''''''''''''''''''
- :ghfile:`bin/numba` - Command line stub, delegates to main in
  ``numba_entry.py``
- :ghfile:`numba/misc/numba_entry.py` - Main function for ``numba`` command line
  tool
- :ghfile:`numba/pycc` - Ahead of time compilation of functions to shared
  library extension
- :ghfile:`numba/pycc/__init__.py` - Main function for ``pycc`` command line
  tool
- :ghfile:`numba/pycc/cc.py` - User-facing API for tagging functions to
  compile ahead of time
- :ghfile:`numba/pycc/compiler.py` - Compiler pipeline for creating
  standalone Python extension modules
- :ghfile:`numba/pycc/llvm_types.py` - Aliases to LLVM data types used by
  ``compiler.py``
- :ghfile:`numba/pycc/modulemixin.c` - C file compiled into every compiled
  extension.  Pulls in C source from Numba core that is needed to make
  extension standalone
- :ghfile:`numba/pycc/platform.py` - Portable interface to platform-specific
  compiler toolchains
- :ghfile:`numba/pycc/decorators.py` - Deprecated decorators for tagging
  functions to compile.  Use ``cc.py`` instead.


CUDA GPU Target
'''''''''''''''

Note that the CUDA target does reuse some parts of the CPU target.

- :ghfile:`numba/cuda/` - The implementation of the CUDA (NVIDIA GPU) target
  and associated unit tests
- :ghfile:`numba/cuda/decorators.py` - Compiler decorators for CUDA kernels
  and device functions
- :ghfile:`numba/cuda/dispatcher.py` - Dispatcher for CUDA JIT functions
- :ghfile:`numba/cuda/printimpl.py` - Special implementation of device printing
- :ghfile:`numba/cuda/libdevice.py` - Registers libdevice functions
- :ghfile:`numba/cuda/kernels/` - Custom kernels for reduction and transpose
- :ghfile:`numba/cuda/device_init.py` - Initializes the CUDA target when
  imported
- :ghfile:`numba/cuda/compiler.py` - Compiler pipeline for CUDA target
- :ghfile:`numba/cuda/intrinsic_wrapper.py` - CUDA device intrinsics
  (shuffle, ballot, etc)
- :ghfile:`numba/cuda/initialize.py` - Deferred initialization of the CUDA
  device and subsystem.  Called only when user imports ``numba.cuda``
- :ghfile:`numba/cuda/simulator_init.py` - Initializes the CUDA simulator
  subsystem (only when user requests it with env var)
- :ghfile:`numba/cuda/random.py` - Implementation of random number generator
- :ghfile:`numba/cuda/api.py` - User facing APIs imported into ``numba.cuda.*``
- :ghfile:`numba/cuda/stubs.py` - Python placeholders for functions that
  only can be used in GPU device code
- :ghfile:`numba/cuda/simulator/` - Simulate execution of CUDA kernels in
  Python interpreter
- :ghfile:`numba/cuda/vectorizers.py` - Subclasses of ufunc/gufunc compilers
  for CUDA
- :ghfile:`numba/cuda/args.py` - Management of kernel arguments, including
  host<->device transfers
- :ghfile:`numba/cuda/target.py` - Typing and target contexts for GPU
- :ghfile:`numba/cuda/cudamath.py` - Type signatures for math functions in
  CUDA Python
- :ghfile:`numba/cuda/errors.py` - Validation of kernel launch configuration
- :ghfile:`numba/cuda/nvvmutils.py` - Helper functions for generating
  NVVM-specific IR
- :ghfile:`numba/cuda/testing.py` - Support code for creating CUDA unit
  tests and capturing standard out
- :ghfile:`numba/cuda/cudadecl.py` - Type signatures of CUDA API (threadIdx,
  blockIdx, atomics) in Python on GPU
- :ghfile:`numba/cuda/cudaimpl.py` - Implementations of CUDA API functions
  on GPU
- :ghfile:`numba/cuda/codegen.py` - Code generator object for CUDA target
- :ghfile:`numba/cuda/cudadrv/` - Wrapper around CUDA driver API
- :ghfile:`numba/cuda/tests/` - CUDA unit tests, skipped when CUDA is not
  detected
- :ghfile:`numba/cuda/tests/cudasim/` - Tests of CUDA simulator
- :ghfile:`numba/cuda/tests/nocuda/` - Tests for NVVM functionality when
  CUDA not present
- :ghfile:`numba/cuda/tests/cudapy/` - Tests of compiling Python functions
  for GPU
- :ghfile:`numba/cuda/tests/cudadrv/` - Tests of Python wrapper around CUDA
  API

