llvm_cbuilder
-------------

A llvm-py Builder wrapper for writing in slightly higher-level constructs.
This is aiming for two usecases:

1. Emit LLVM code in a more human-readable way;

2. Writing low-level code that you can't do it properly/portably with C, e.g
   template (generic), atomic operations, memory ordering...



Parallel Vectorize
------------------

`parallel_vectorize.py` implements a set of code generator that bases on
llvm_cbuilder to create specialized parallel ufunc.

See `test_parallel_vectorize_numpy*.py` for testing and demo of the code
with the new `numpy.fromfunc()`.


