.. _roadmap:

*******************
Numba Roadmap
*******************

* zero-cost exceptions
    - stack trace through libunwind/apple backtrace/LLVM info based on instruction pointer
* debug info
* native dispatch (SEP 200/201)
* Extension classes with autojitting methods
    - based on native dispatch, replaces vtable
    - similar approach for native attributes (dynamic storage/data polymorphism)
* struct references
    - use cheap heap allocated objects + garbage collection?
    - use julia for the lifting
* blaze support
    - compile abstract blaze expressions into kernels
    - generate native call to blaze kernel
* code caching
* generators/parallel tasks

