
Glossary
========

.. glossary::

   ahead-of-time compilation
   AOT compilation
   AOT
      Compilation of a function in a separate step before running the
      program code, producing an on-disk binary object which can be distributed
      independently.  This is the traditional kind of compilation known
      in languages such as C, C++ or Fortran.

   compile-time constant
      An expression whose value Numba can infer and freeze at compile-time.
      Global variables and closure variables are compile-time constants.

   just-in-time compilation
   JIT compilation
   JIT
      Compilation of a function at execution time, as opposed to
      :term:`ahead-of-time compilation`.

   JIT function
      Shorthand for "a function :term:`JIT-compiled <JIT>` with Numba using
      the :ref:`@jit <jit>` decorator."

   loop-lifting
      See :term:`loop-jitting`.

   loop-jitting
      A feature of compilation in :term:`object mode` where a loop can be
      automatically extracted and compiled in :term:`nopython mode`.  This
      allows functions with operations unsupported in nopython mode (such as
      array creation) to see significant performance improvements if they
      contain loops with only nopython-supported operations.

   nopython mode
      A Numba compilation mode that generates code that does not access the
      Python C API.  This compilation mode produces the highest performance
      code, but requires that the native types of all values in the function
      can be :term:`inferred <type inference>`.  Unless otherwise instructed,
      the ``@jit`` decorator will automatically fall back to :term:`object
      mode` if nopython mode cannot be used.

   object mode
      A Numba compilation mode that generates code that handles all values
      as Python objects and uses the Python C API to perform all operations
      on those objects.  Code compiled in object mode will often run
      no faster than Python interpreted code, unless the Numba compiler can
      take advantage of :term:`loop-jitting`.

   type inference
      The process by which Numba determines the specialized types of all
      values within a function being compiled.  Type inference can fail
      if arguments or globals have Python types unknown to Numba, or if
      functions are used that are not recognized by Numba.  Sucessful
      type inference is a prerequisite for compilation in
      :term:`nopython mode`.

   ufunc
      A NumPy `universal function <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.
      Numba can create new compiled ufuncs with
      the :ref:`@vectorize <vectorize>` decorator.
