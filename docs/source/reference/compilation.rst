Compilation
===========


.. decorator:: numba.jit([signature], *, nopython=False, nojit=False, forceobj=False, locals={})

   Compile the decorated function on-the-fly to produce efficient machine
   code.  All parameters all optional.

   *signature* represents the expected :ref:`numba-types` of function arguments
   and return value.  It can be given in several forms:

   * A tuple of :ref:`numba-types` arguments (for example
     ``(numba.int32, numba.double)``) representing the types of the
     function's arguments; Numba will then infer an appropriate return
     type from the arguments.
   * A call signature using :ref:`numba-types`, specifying both return
     type and argument types. This can be given in intuitive form
     (for example ``numba.void(numba.int32, numba.double)``).
   * A string representation of one of the above, for example
     ``"void(int32, double)"``.

   *nopython* and *nojit* are boolean flags.  *locals* is a mapping of
   local variable names to :ref:`numba-types`.

   This decorator has several modes of operation:

   * If *signature* is given, a single specialization is compiled
     corresponding to this signature.  Calling the decorated function will
     then try to convert the arguments to this signature, and raise a
     :class:`TypeError` if converting fails.  If converting succeeds, the
     compiled machine code is executed with the converted arguments and the
     return value is converted back according to the signature.

   * If *signature* is not given, the decorated function implements
     multiple dispatch and lazy compilation.  Each call to the decorated
     function will try to re-use an existing specialization if it exists
     (for example, a call with two integer arguments may re-use a
     specialization for argument types ``(numba.int64, numba.int64)``).
     If no suitable specialization exists, a new specialization is compiled
     on-the-fly, stored for later use, and executed with the converted
     arguments.

   If true, *nopython* forces the function to be compiled in :term:`nopython
   mode`. If not possible, compilation will raise an error.

   If true, *forceobj* forces the function to be compiled in :term:`object
   mode`.  Since object mode is slower than nopython mode, this is mostly
   useful for testing purposes.

   If true, *nogil* tries to release the :py:term:`global interpreter lock`
   inside the compiled function.

   The *locals* dictionary may be used to force the :ref:`numba-types`
   of particular local variables, for example if you want to force the
   use of single precision floats at some point.  In general, we recommend
   you let Numba's compiler infer the types of local variables by itself.

   Not putting any parentheses after the decorator is equivalent to calling
   the decorator without any arguments, i.e.::

      @jit
      def f(x): ...

   is equivalent to::

      @jit()
      def f(x): ...

   .. note::
      If no *signature* is given, compilation errors will be raised when
      the actual compilation occurs, i.e. when the function is first called
      with some given argument types.

   .. note::
      Compilation can be influenced by some dedicated :ref:`numba-envvars`.


.. decorator:: numba.vectorize(signatures, *, ...)

.. decorator:: numba.guvectorize(...)


.. todo:: write this
