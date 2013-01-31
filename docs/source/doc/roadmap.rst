.. _roadmap:

*******************
Numba Roadmap
*******************

* full support for type inference on NumPy functions
* compile everything -- don't fail unless (nopython option given)
* speeding up calling overhead of autojit
* autojit of classes
  * based on native dispatch (replace vtable)
  * similar approach for native attributes (dynamic storage / data polymorphism)
* Better code-generated for loops (using TBAA or other code-generation) to avoid multiple stride-array lookups.
* zero-cost exceptions
    - stack trace through libunwind/apple backtrace/LLVM info based on instruction pointer
* debug info
* native dispatch (SEP 200/201)
* vector-types in Numba
* struct references
    - use cheap heap allocated objects + garbage collection?
    - use julia for the lifting
* blaze support
    - compile abstract blaze expressions into kernels
    - generate native call to blaze kernel
* code caching
* generators/parallel tasks
* typed containers (dict, list, tuple)
* Python 3.3 support (optional type-annotations)
* SPIR support (OpenCL)
* GPU support in Numba
* Array Expression support in Numba
