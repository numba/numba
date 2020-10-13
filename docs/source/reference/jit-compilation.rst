Just-in-Time compilation
========================


JIT functions
-------------

.. _jit-decorator:

.. decorator:: numba.jit(signature=None, nopython=False, nogil=False, cache=False, forceobj=False, parallel=False, error_model='python', fastmath=False, locals={}, boundscheck=False)

   Compile the decorated function on-the-fly to produce efficient machine
   code.  All parameters are optional.

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

   *nopython* and *nogil* are boolean flags.  *locals* is a mapping of
   local variable names to :ref:`numba-types`.

   This decorator has several modes of operation:

   * If one or more signatures are given in *signature*, a specialization is
     compiled for each of them.  Calling the decorated function will then try
     to choose the best matching signature, and raise a :class:`TypeError` if
     no appropriate conversion is available for the function arguments.  If
     converting succeeds, the compiled machine code is executed with the
     converted arguments and the return value is converted back according to
     the signature.

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
   compile the function in :term:`nopython mode`, otherwise a compilation
   warning will be printed.

   .. _jit-decorator-cache:

   If true, *cache* enables a file-based cache to shorten compilation times
   when the function was already compiled in a previous invocation.
   The cache is maintained in the ``__pycache__`` subdirectory of
   the directory containing the source file; if the current user is not
   allowed to write to it, though, it falls back to a platform-specific
   user-wide cache directory (such as ``$HOME/.cache/numba`` on Unix
   platforms).

   .. _jit-decorator-parallel:

   If true, *parallel* enables the automatic parallelization of a number of
   common Numpy constructs as well as the fusion of adjacent parallel
   operations to maximize cache locality.

   The *error_model* option controls the divide-by-zero behavior.
   Setting it to 'python' causes divide-by-zero to raise exception like CPython.
   Setting it to 'numpy' causes divide-by-zero to set the result to *+/-inf* or
   *nan*.

   Not all functions can be cached, since some functionality cannot be
   always persisted to disk.  When a function cannot be cached, a
   warning is emitted.

   .. _jit-decorator-fastmath:

   If true, *fastmath* enables the use of otherwise unsafe floating point
   transforms as described in the
   `LLVM documentation <https://llvm.org/docs/LangRef.html#fast-math-flags>`_.
   Further, if :ref:`Intel SVML <intel-svml>` is installed faster but less
   accurate versions of some math intrinsics are used (answers to within
   ``4 ULP``).

   .. _jit-decorator-boundscheck:

   If True, ``boundscheck`` enables bounds checking for array indices. Out of
   bounds accesses will raise IndexError. The default is to not do bounds
   checking. If bounds checking is disabled, out of bounds accesses can
   produce garbage results or segfaults. However, enabling bounds checking
   will slow down typical functions, so it is recommended to only use this
   flag for debugging. You can also set the `NUMBA_BOUNDSCHECK` environment
   variable to 0 or 1 to globally override this flag.

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

   The decorator returns a :class:`Dispatcher` object.

   .. note::
      If no *signature* is given, compilation errors will be raised when
      the actual compilation occurs, i.e. when the function is first called
      with some given argument types.

   .. note::
      Compilation can be influenced by some dedicated :ref:`numba-envvars`.


Generated JIT functions
-----------------------

.. decorator:: numba.generated_jit(nopython=False, nogil=False, cache=False, forceobj=False, locals={})

   Like the :func:`~numba.jit` decorator, but calls the decorated function at
   compile-time, passing the *types* of the function's arguments.
   The decorated function must return a callable which will be compiled as
   the function's implementation for those types, allowing flexible kinds of
   specialization.

   The :func:`~numba.generated_jit` decorator returns a :class:`Dispatcher` object.


Dispatcher objects
------------------

.. class:: Dispatcher

   The class of objects created by calling :func:`~numba.jit` or
   :func:`~numba.generated_jit`.  You shouldn't try to create such an object
   in any other way.  Calling a Dispatcher object calls the compiled
   specialization for the arguments with which it is called, letting it
   act as an accelerated replacement for the Python function which was compiled.

   In addition, Dispatcher objects have the following methods and attributes:

   .. attribute:: py_func

      The pure Python function which was compiled.

   .. method:: inspect_types(file=None, pretty=False)

      Print out a listing of the function source code annotated line-by-line
      with the corresponding Numba IR, and the inferred types of the various
      variables.  If *file* is specified, printing is done to that file
      object, otherwise to sys.stdout. If *pretty* is set to True then colored
      ANSI will be produced in a terminal and HTML in a notebook.

      .. seealso:: :ref:`architecture`

   .. method:: inspect_llvm(signature=None)

      Return a dictionary keying compiled function signatures to the human
      readable LLVM IR generated for the function.  If the signature
      keyword is specified a string corresponding to that individual
      signature is returned.

   .. method:: inspect_asm(signature=None)

      Return a dictionary keying compiled function signatures to the
      human-readable native assembly code for the function.  If the
      signature keyword is specified a string corresponding to that
      individual signature is returned.

   .. method:: inspect_cfg(signature=None, show_wrapped)

      Return a dictionary keying compiled function signatures to the
      control-flow graph objects for the function.  If the signature keyword is
      specified a string corresponding to that individual signature is returned.

      The control-flow graph objects can be stringified (``str`` or ``repr``)
      to get the textual representation of the graph in DOT format.  Or, use
      its ``.display(filename=None, view=False)`` method to plot the graph.
      The *filename* option can be set to a specific path for the rendered
      output to write to.  If *view* option is True, the plot is opened by
      the system default application for the image format (PDF). In IPython
      notebook, the returned object can be plot inlined.

      Usage::

        @jit
        def foo():
          ...

        # opens the CFG in system default application
        foo.inspect_cfg(foo.signatures[0]).display(view=True)


   .. method:: inspect_disasm_cfg(signature=None)

      Return a dictionary keying compiled function signatures to the
      control-flow graph of the disassembly of the underlying compiled ``ELF``
      object.  If the signature keyword is specified a control-flow graph
      corresponding to that individual signature is returned. This function is
      execution environment aware and will produce SVG output in Jupyter
      notebooks and ASCII in terminals.

      Example::

        @njit
        def foo(x):
            if x < 3:
                return x + 1
            return x + 2

        foo(10)

        print(foo.inspect_disasm_cfg(signature=foo.signatures[0]))

      Gives::

        [0x08000040]>  # method.__main__.foo_241_long_long (int64_t arg1, int64_t arg3);
         ─────────────────────────────────────────────────────────────────────┐
        │  0x8000040                                                          │
        │ ; arg3 ; [02] -r-x section size 279 named .text                     │
        │   ;-- section..text:                                                │
        │   ;-- .text:                                                        │
        │   ;-- __main__::foo$241(long long):                                 │
        │   ;-- rip:                                                          │
        │ 25: method.__main__.foo_241_long_long (int64_t arg1, int64_t arg3); │
        │ ; arg int64_t arg1 @ rdi                                            │
        │ ; arg int64_t arg3 @ rdx                                            │
        │ ; 2                                                                 │
        │ cmp rdx, 2                                                          │
        │ jg 0x800004f                                                        │
        └─────────────────────────────────────────────────────────────────────┘
                f t
                │ │
                │ └──────────────────────────────┐
                └──┐                             │
                   │                             │
            ┌─────────────────────────┐   ┌─────────────────────────┐
            │  0x8000046              │   │  0x800004f              │
            │ ; arg3                  │   │ ; arg3                  │
            │ inc rdx                 │   │ add rdx, 2              │
            │ ; arg3                  │   │ ; arg3                  │
            │ mov qword [rdi], rdx    │   │ mov qword [rdi], rdx    │
            │ xor eax, eax            │   │ xor eax, eax            │
            │ ret                     │   │ ret                     │
            └─────────────────────────┘   └─────────────────────────┘

   .. method:: recompile()

      Recompile all existing signatures.  This can be useful for example if
      a global or closure variable was frozen by your function and its value
      in Python has changed.  Since compiling isn't cheap, this is mainly
      for testing and interactive use.

   .. method:: parallel_diagnostics(signature=None, level=1)

      Print parallel diagnostic information for the given signature. If no
      signature is present it is printed for all known signatures. ``level`` is
      used to adjust the verbosity, ``level=1`` (default) is minimum verbosity,
      levels 2, 3, and 4 provide increasing levels of verbosity.

   .. method:: get_metadata(signature=None)

      Obtain the compilation metadata for a given signature. This is useful for
      developers of Numba and Numba extensions.


Vectorized functions (ufuncs and DUFuncs)
-----------------------------------------

.. decorator:: numba.vectorize(*, signatures=[], identity=None, nopython=True, target='cpu', forceobj=False, cache=False, locals={})

   Compile the decorated function and wrap it either as a `Numpy
   ufunc`_ or a Numba :class:`~numba.DUFunc`.  The optional
   *nopython*, *forceobj* and *locals* arguments have the same meaning
   as in :func:`numba.jit`.

   *signatures* is an optional list of signatures expressed in the
   same form as in the :func:`numba.jit` *signature* argument.  If
   *signatures* is non-empty, then the decorator will compile the user
   Python function into a Numpy ufunc.  If no *signatures* are given,
   then the decorator will wrap the user Python function in a
   :class:`~numba.DUFunc` instance, which will compile the user
   function at call time whenever Numpy can not find a matching loop
   for the input arguments.  *signatures* is required if *target* is
   ``"parallel"``.

   *identity* is the identity (or unit) value of the function being
   implemented.  Possible values are 0, 1, None, and the string
   ``"reorderable"``.  The default is None.  Both None and
   ``"reorderable"`` mean the function has no identity value;
   ``"reorderable"`` additionally specifies that reductions along multiple
   axes can be reordered.

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

   *target* is a string for backend target; Available values are "cpu",
   "parallel", and "cuda".  To use a multithreaded version, change the
   target to "parallel" (which requires signatures to be specified)::

      @vectorize(["float64(float64)", "float32(float32)"], target='parallel')
      def f(x): ...

   For the CUDA target, use "cuda"::

      @vectorize(["float64(float64)", "float32(float32)"], target='cuda')
      def f(x): ...

   The compiled function can be cached to reduce future compilation time.
   It is enabled by setting *cache* to True. Only the "cpu" and "parallel"
   targets support caching.


.. decorator:: numba.guvectorize(signatures, layout, *, identity=None, nopython=True, target='cpu', forceobj=False, cache=False, locals={})

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
   in the layout string (e.g. ``"(n),(n)"``). When doing this, it is important
   to be aware that changes to the input arrays cannot always be relied on to be
   visible outside the execution of the ufunc, as NumPy may pass in temporary
   arrays as inputs (for example, if a cast is required).

   .. seealso::
      Specification of the `layout string <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html#details-of-signature>`_
      as supported by Numpy.  Note that Numpy uses the term "signature",
      which we unfortunately use for something else.

   The compiled function can be cached to reduce future compilation time.
   It is enabled by setting *cache* to True. Only the "cpu" and "parallel"
   targets support caching.

.. _Numpy ufunc: http://docs.scipy.org/doc/numpy/reference/ufuncs.html

.. class:: numba.DUFunc

   The class of objects created by calling :func:`numba.vectorize`
   with no signatures.

   DUFunc instances should behave similarly to Numpy
   :class:`~numpy.ufunc` objects with one important difference:
   call-time loop generation.  When calling a ufunc, Numpy looks at
   the existing loops registered for that ufunc, and will raise a
   :class:`~python.TypeError` if it cannot find a loop that it cannot
   safely cast the inputs to suit.  When calling a DUFunc, Numba
   delegates the call to Numpy.  If the Numpy ufunc call fails, then
   Numba attempts to build a new loop for the given input types, and
   calls the ufunc again.  If this second call attempt fails or a
   compilation error occurs, then DUFunc passes along the exception to
   the caller.

   .. seealso::

      The ":ref:`dynamic-universal-functions`" section in the user's
      guide demonstrates the call-time behavior of
      :class:`~numba.DUFunc`, and discusses the impact of call order
      on how Numba generates the underlying :class:`~numpy.ufunc`.

   .. attribute:: ufunc

      The actual Numpy :class:`~numpy.ufunc` object being built by the
      :class:`~numba.DUFunc` instance.  Note that the
      :class:`~numba.DUFunc` object maintains several important data
      structures required for proper ufunc functionality (specifically
      the dynamically compiled loops).  Users should not pass the
      :class:`~numpy.ufunc` value around without ensuring the
      underlying :class:`~numba.DUFunc` will not be garbage collected.

   .. attribute:: nin

      The number of DUFunc (ufunc) inputs.  See `ufunc.nin`_.

   .. attribute:: nout

      The number of DUFunc outputs.  See `ufunc.nout`_.

   .. attribute:: nargs

      The total number of possible DUFunc arguments (should be
      :attr:`~numba.DUFunc.nin` + :attr:`~numba.DUFunc.nout`).
      See `ufunc.nargs`_.

   .. attribute:: ntypes

      The number of input types supported by the DUFunc.  See
      `ufunc.ntypes`_.

   .. attribute:: types

      A list of the supported types given as strings.  See
      `ufunc.types`_.

   .. attribute:: identity

      The identity value when using the ufunc as a reduction.  See
      `ufunc.identity`_.

   .. method:: reduce(A, *, axis, dtype, out, keepdims)

      Reduces *A*\'s dimension by one by applying the DUFunc along one
      axis.  See `ufunc.reduce`_.

   .. method:: accumulate(A, *, axis, dtype, out)

      Accumulate the result of applying the operator to all elements.
      See `ufunc.accumulate`_.

   .. method:: reduceat(A, indices, *, axis, dtype, out)

      Performs a (local) reduce with specified slices over a single
      axis.  See `ufunc.reduceat`_.

   .. method:: outer(A, B)

      Apply the ufunc to all pairs (*a*, *b*) with *a* in *A*, and *b*
      in *B*.  See `ufunc.outer`_.

   .. method:: at(A, indices, *, B)

      Performs unbuffered in place operation on operand *A* for
      elements specified by *indices*.  If you are using Numpy 1.7 or
      earlier, this method will not be present.  See `ufunc.at`_.


.. note::
   Vectorized functions can, in rare circumstances, show
   :ref:`unexpected warnings or errors <ufunc-fpu-errors>`.


.. _`ufunc.nin`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.nin.html#numpy.ufunc.nin

.. _`ufunc.nout`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.nout.html#numpy.ufunc.nout

.. _`ufunc.nargs`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.nargs.html#numpy.ufunc.nargs

.. _`ufunc.ntypes`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.ntypes.html#numpy.ufunc.ntypes

.. _`ufunc.types`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.types.html#numpy.ufunc.types

.. _`ufunc.identity`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.identity.html#numpy.ufunc.identity

.. _`ufunc.reduce`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduce.html#numpy.ufunc.reduce

.. _`ufunc.accumulate`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.accumulate.html#numpy.ufunc.accumulate

.. _`ufunc.reduceat`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.reduceat.html#numpy.ufunc.reduceat

.. _`ufunc.outer`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.outer.html#numpy.ufunc.outer

.. _`ufunc.at`: http://docs.scipy.org/doc/numpy/reference/generated/numpy.ufunc.at.html#numpy.ufunc.at


C callbacks
-----------

.. decorator:: numba.cfunc(signature, nopython=False, cache=False, locals={})

   Compile the decorated function on-the-fly to produce efficient machine
   code.  The compiled code is wrapped in a thin C callback that makes it
   callable using the natural C ABI.

   The *signature* is a single signature representing the signature of the
   C callback.  It must have the same form as in :func:`~numba.jit`.
   The decorator does not check that the types in the signature have
   a well-defined representation in C.

   *nopython* and *cache* are boolean flags.  *locals* is a mapping of
   local variable names to :ref:`numba-types`.  They all have the same
   meaning as in :func:`~numba.jit`.

   The decorator returns a :class:`CFunc` object.

   .. note::
      C callbacks currently do not support :term:`object mode`.


.. class:: CFunc

   The class of objects created by :func:`~numba.cfunc`.  :class:`CFunc`
   objects expose the following attributes and methods:

   .. attribute:: address

      The address of the compiled C callback, as an integer.

   .. attribute:: cffi

      A `cffi`_ function pointer instance, to be passed as an argument to
      `cffi`_-wrapped functions.  The pointer's type is ``void *``, so
      only minimal type checking will happen when passing it to `cffi`_.

   .. attribute:: ctypes

      A :mod:`ctypes` callback instance, as if it were created using
      :func:`ctypes.CFUNCTYPE`.

   .. attribute:: native_name

      The name of the compiled C callback.

   .. method:: inspect_llvm()

      Return the human-readable LLVM IR generated for the C callback.
      :attr:`native_name` is the name under which this callback is defined
      in the IR.


.. _cffi: https://cffi.readthedocs.org/
