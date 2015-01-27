
Overview
========

Numba gives you the power to speed up your applications with high performance
functions written directly in Python.

Numba generates optimized machine code from pure Python code using
the `LLVM compiler infrastructure <http://llvm.org/>`_.  With a few simple
annotations, array-oriented and math-heavy Python code can be
just-in-time optimized to performance similar as C, C++ and Fortran, without
having to switch languages or Python interpreters.

Numba's main features are:

* :ref:`on-the-fly code generation <jit>` (at import time or runtime, at the
  user's preference)
* native code generation for the CPU (default) and
  :doc:`GPU hardware </cuda/index>`
* integration with the Python scientific software stack (thanks to Numpy)

Here is how a Numba-optimized function, taking a Numpy array as argument,
might look like::

   @numba.jit
   def sum2d(arr):
       M, N = arr.shape
       result = 0.0
       for i in range(M):
           for j in range(N):
               result += arr[i,j]
       return result

