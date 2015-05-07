Compilation
===========

JIT functions
-------------

.. decorator:: numba.jit(signature=None, nopython=False, nogil=False, forceobj=False, locals={})

   Compile the decorated function on-the-fly to produce efficient machine
   code.  All parameters all optional.

   If present, the *signature* is either a single signature or a list of
   signatures representing the expected :ref:`numba-types` of function
   arguments and return values.  Each signature can be given in several
   forms:

   * A tuple of :ref:`numba-types` arguments (for example
     ``(numba.int32, numba.double)``) representing the types of the
     function's arguments; Numba will then infer an appropriate return
     type from the arguments.
   * A call signature using :ref:`numba-types`, specifying both return
     type and argument types. This can be given in intuitive form
     (for example ``numba.void(numba.int32, numba.double)``).
   * A string representation of one of the above, for example
     ``"void(int32, double)"``.  All type names used in the string are assumed
     to be defined in the ``numba.types`` module.

   *nopython* and *nojit* are boolean flags.  *locals* is a mapping of
   local variable names to :ref:`numba-types`.

   This decorator has several modes of operation:

   * If one more *signature* is given, a specialization is compiled
     for each signature.  Calling the decorated function will
     then try to choose the best matching signature, and raise a
     :class:`TypeError` if no appropriate conversion is available for the
     funciton arguments.  If converting succeeds, the compiled machine code
     is executed with the converted arguments and the return value is
     converted back according to the signature.

   * If no *signature* is given, the decorated function implements
     lazy compilation.  Each call to the decorated function will try to
     re-use an existing specialization if it exists (for example, a call
     with two integer arguments may re-use a specialization for argument
     types ``(numba.int64, numba.int64)``).  If no suitable specialization
     exists, a new specialization is compiled on-the-fly, stored for later
     use, and executed with the converted arguments.

   If true, *nopython* forces the function to be compiled in :term:`nopython
   mode`. If not possible, compilation will raise an error.

   If true, *forceobj* forces the function to be compiled in :term:`object
   mode`.  Since object mode is slower than nopython mode, this is mostly
   useful for testing purposes.

   If true, *nogil* tries to release the :py:term:`global interpreter lock`
   inside the compiled function.  The GIL will only be released if Numba can
   compile the function in :term:`object mode`, otherwise a compilation
   warning will be printed.

   The *locals* dictionary may be used to force the :ref:`numba-types`
   of particular local variables, for example if you want to force the
   use of single precision floats at some point.  In general, we recommend
   you let Numba's compiler infer the types of local variables by itself.

   Here is an example with two signatures::

      @jit(["int32(int32)", "float32(float32)"], nopython=True)
      def f(x): ...

   Not putting any parentheses after the decorator is equivalent to calling
   the decorator without any arguments, i.e.::

      @jit
      def f(x): ...

   is equivalent to::

      @jit()
      def f(x): ...

   The decorator returns a Dispatcher object.

   .. note::
      If no *signature* is given, compilation errors will be raised when
      the actual compilation occurs, i.e. when the function is first called
      with some given argument types.

   .. note::
      Compilation can be influenced by some dedicated :ref:`numba-envvars`.


.. class:: Dispatcher

   The class of objects created by calling :func:`numba.jit`.  You shouldn't
   try to create such an object in any other way.  Dispatcher objects have
   the following methods and attributes:

   .. attribute:: py_func

      The pure Python function which was compiled.

   .. method:: inspect_types(file=None)

      Print out a listing of the function source code annotated line-by-line
      with the corresponding Numba IR, and the inferred types of the various
      variables.  If *file* is specified, printing is done to that file
      object, otherwise to sys.stdout.

      .. seealso:: :ref:`architecture`

   .. method:: inspect_llvm(signature=None)

      Return a dictionary keying compiled function signatures to the human 
      readable LLVM IR generated for the function.  If the signature 
      keyword is specified a string corresponding to that individual 
      signature is returned.  

   .. method:: inspect_asm(signature=None)

      Return a dictionary keying compiled function signatures to the
      human-readable native assembler code for the function.  If the 
      signature keyword is specified a string corresponding to that 
      individual signature is returned.  

   .. method:: recompile()

      Recompile all existing signatures.  This can be useful for example if
      a global or closure variable was frozen by your function and its value
      in Python has changed.  Since compiling isn't cheap, this is mainly
      for testing and interactive use.


Vectorized functions (ufuncs)
-----------------------------

.. decorator:: numba.vectorize(signatures, *, identity=None, nopython=True, forceobj=False, locals={})

   Compile the decorated function on-the-fly and wrap it as a
   `Numpy ufunc`_.  The optional *nopython*, *forceobj* and
   *locals* arguments have the same meaning as in :func:`numba.jit`.

   *signatures* is a mandatory list of signatures expressed in the same
   form as in the :func:`numba.jit` *signature* argument.

   *identity* is the identity (or unit) value of the function being
   implemented.  Possible values are 0, 1, None, and the string
   ``"reorderable"``.  The default is None.  Both None and
   ``"reorderable"`` mean the function has no identity value;
   ``"reorderable"`` additionally specifies that reductions along multiple
   axes can be reordered.  (Note that ``"reorderable"`` is only supported in
   Numpy 1.7 or later.)

   If there are several *signatures*, they must be ordered from the more
   specific to the least specific.  Otherwise, Numpy's type-based
   dispatching may not work as expected.  For example, the following is
   wrong::

      @vectorize(["float64(float64)", "float32(float32)"])
      def f(x): ...

   as running it over a single-precision array will choose the ``float64``
   version of the compiled function, leading to much less efficient
   execution.  The correct invocation is::

      @vectorize(["float32(float32)", "float64(float64)"])
      def f(x): ...


.. decorator:: numba.guvectorize(signatures, layout, *, identity=None, nopython=True, forceobj=False, locals={})

   Generalized version of :func:`numba.vectorize`.  While
   :func:`numba.vectorize` will produce a simple ufunc whose core
   functionality (the function you are decorating) operates on scalar
   operands and returns a scalar value, :func:`numba.guvectorize`
   allows you to create a `Numpy ufunc`_ whose core function takes array
   arguments of various dimensions.

   The additional argument *layout* is a string specifying, in symbolic
   form, the dimensionality and size relationship of the argument types
   and return types.  For example, a matrix multiplication will have
   a layout string of ``"(m,n),(n,p)->(m,p)"``.  Its definition might
   be (function body omitted)::

      @guvectorize(["void(float64[:,:], float64[:,:], float64[:,:])"],
                   "(m,n),(n,p)->(m,p)")
      def f(a, b, result):
          """Fill-in *result* matrix such as result := a * b"""
          ...

   If one of the arguments should be a scalar, the corresponding layout
   specification is ``()`` and the argument will really be given to
   you as a zero-dimension array (you have to dereference it to get the
   scalar value).  For example, a :ref:`one-dimension moving average <example-movemean>`
   with a parameterable window width may have a layout string of ``"(n),()->(n)"``.

   Note that any output will be given to you preallocated as an additional
   function argument: your code has to fill it with the appropriate values
   for the function you are implementing.

   If your function doesn't take an output array, you should omit the "arrow"
   in the layout string (e.g. ``"(n),(n)"``).

   .. seealso::
      Specification of the `layout string <http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html#details-of-signature>`_
      as supported by Numpy.  Note that Numpy uses the term "signature",
      which we unfortunately use for something else.


.. _Numpy ufunc: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
