
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

   bytecode
   Python bytecode
      The original form in which Python functions are executed.  Python
      bytecode describes a stack-machine executing abstract (untyped)
      operations using operands from both the function stack and the
      execution environment (e.g. global variables).

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
   loop-jitting
      A feature of compilation in :term:`object mode` where a loop can be
      automatically extracted and compiled in :term:`nopython mode`.  This
      allows functions with operations unsupported in nopython mode to see
      significant performance improvements if they contain loops with only
      nopython-supported operations.

   lowering
      The act of translating :term:`Numba IR` into LLVM IR.  The term
      "lowering" stems from the fact that LLVM IR is low-level and
      machine-specific while Numba IR is high-level and abstract.

   NPM
   nopython mode
      A Numba compilation mode that generates code that does not access the
      Python C API.  This compilation mode produces the highest performance
      code, but requires that the native types of all values in the function
      can be :term:`inferred <type inference>`.  Unless otherwise instructed,
      the ``@jit`` decorator will automatically fall back to :term:`object
      mode` if nopython mode cannot be used.

   Numba IR
   Numba intermediate representation
      A representation of a piece of Python code which is more amenable
      to analysis and transformations than the original Python
      :term:`bytecode`.

   object mode
      A Numba compilation mode that generates code that handles all values
      as Python objects and uses the Python C API to perform all operations
      on those objects.  Code compiled in object mode will often run
      no faster than Python interpreted code, unless the Numba compiler can
      take advantage of :term:`loop-jitting`.

   ``OptionalType``
     An ``OptionalType`` is effectively a type union of a ``type`` and ``None``.
     They typically occur in practice due to a variable being set to ``None``
     and then in a branch the variable being set to some other value. It's
     often not possible at compile time to determine if the branch will execute
     so to permit :term:`type inference` to complete, the type of the variable
     becomes the union of a ``type`` (from the value) and ``None``,
     i.e. ``OptionalType(type)``.

   type inference
      The process by which Numba determines the specialized types of all
      values within a function being compiled.  Type inference can fail
      if arguments or globals have Python types unknown to Numba, or if
      functions are used that are not recognized by Numba.  Successful
      type inference is a prerequisite for compilation in
      :term:`nopython mode`.

   typing
      The act of running :term:`type inference` on a value or operation.

   ufunc
      A NumPy `universal function <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.
      Numba can create new compiled ufuncs with
      the :ref:`@vectorize <vectorize>` decorator.

   reflection
      In numba, when a mutable container is passed as argument to a nopython
      function from the Python interpreter, the container object and all its
      contained elements are converted into nopython values.  To match the
      semantics of Python, any mutation on the container inside the nopython
      function must be visible in the Python interpreter.  To do so, Numba
      must update the container and its elements and convert them back into
      Python objects during the transition back into the interpreter.

      Not to be confused with Python's "reflection" in the context of binary
      operators (see https://docs.python.org/3.5/reference/datamodel.html).
