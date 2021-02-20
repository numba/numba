==================================
Creating NumPy universal functions
==================================

There are two types of universal functions:

* Those which operate on scalars, these are "universal functions" or *ufuncs*
  (see ``@vectorize`` below).
* Those which operate on higher dimensional arrays and scalars, these are
  "generalized universal functions" or *gufuncs* (``@guvectorize`` below).

.. _vectorize:

The ``@vectorize`` decorator
============================

Numba's vectorize allows Python functions taking scalar input arguments to
be used as NumPy `ufuncs`_.  Creating a traditional NumPy ufunc is
not the most straightforward process and involves writing some C code.
Numba makes this easy.  Using the :func:`~numba.vectorize` decorator, Numba
can compile a pure Python function into a ufunc that operates over NumPy
arrays as fast as traditional ufuncs written in C.

.. _ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html

Using :func:`~numba.vectorize`, you write your function as operating over
input scalars, rather than arrays.  Numba will generate the surrounding
loop (or *kernel*) allowing efficient iteration over the actual inputs.

The :func:`~numba.vectorize` decorator has two modes of operation:

* Eager, or decoration-time, compilation: If you pass one or more type
  signatures to the decorator, you will be building a Numpy universal
  function (ufunc).  The rest of this subsection describes building
  ufuncs using decoration-time compilation.

* Lazy, or call-time, compilation: When not given any signatures, the
  decorator will give you a Numba dynamic universal function
  (:class:`~numba.DUFunc`) that dynamically compiles a new kernel when
  called with a previously unsupported input type.  A later
  subsection, ":ref:`dynamic-universal-functions`", describes this mode in
  more depth.

As described above, if you pass a list of signatures to the
:func:`~numba.vectorize` decorator, your function will be compiled
into a Numpy ufunc.  In the basic case, only one signature will be
passed::

   from numba import vectorize, float64

   @vectorize([float64(float64, float64)])
   def f(x, y):
       return x + y

If you pass several signatures, beware that you have to pass most specific
signatures before least specific ones (e.g., single-precision floats
before double-precision floats), otherwise type-based dispatching will not work
as expected::

   @vectorize([int32(int32, int32),
               int64(int64, int64),
               float32(float32, float32),
               float64(float64, float64)])
   def f(x, y):
       return x + y

The function will work as expected over the specified array types::

   >>> a = np.arange(6)
   >>> f(a, a)
   array([ 0,  2,  4,  6,  8, 10])
   >>> a = np.linspace(0, 1, 6)
   >>> f(a, a)
   array([ 0. ,  0.4,  0.8,  1.2,  1.6,  2. ])

but it will fail working on other types::

   >>> a = np.linspace(0, 1+1j, 6)
   >>> f(a, a)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: ufunc 'ufunc' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


You might ask yourself, "why would I go through this instead of compiling
a simple iteration loop using the :ref:`@jit <jit>` decorator?".  The
answer is that NumPy ufuncs automatically get other features such as
reduction, accumulation or broadcasting.  Using the example above::

   >>> a = np.arange(12).reshape(3, 4)
   >>> a
   array([[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11]])
   >>> f.reduce(a, axis=0)
   array([12, 15, 18, 21])
   >>> f.reduce(a, axis=1)
   array([ 6, 22, 38])
   >>> f.accumulate(a)
   array([[ 0,  1,  2,  3],
          [ 4,  6,  8, 10],
          [12, 15, 18, 21]])
   >>> f.accumulate(a, axis=1)
   array([[ 0,  1,  3,  6],
          [ 4,  9, 15, 22],
          [ 8, 17, 27, 38]])

.. seealso::
   `Standard features of ufuncs <http://docs.scipy.org/doc/numpy/reference/ufuncs.html#ufunc>`_ (NumPy documentation).

.. note::
   Only the broadcasting features of ufuncs are supported in compiled code.

The :func:`~numba.vectorize` decorator supports multiple ufunc targets:

=================       ===============================================================
Target                    Description
=================       ===============================================================
cpu                     Single-threaded CPU


parallel                Multi-core CPU


cuda                    CUDA GPU

                        .. NOTE:: This creates an *ufunc-like* object.
			  See `documentation for CUDA ufunc <../cuda/ufunc.html>`_ for detail.
=================       ===============================================================

A general guideline is to choose different targets for different data sizes
and algorithms.
The "cpu" target works well for small data sizes (approx. less than 1KB) and low
compute intensity algorithms. It has the least amount of overhead.
The "parallel" target works well for medium data sizes (approx. less than 1MB).
Threading adds a small delay.
The "cuda" target works well for big data sizes (approx. greater than 1MB) and
high compute intensity algorithms.  Transferring memory to and from the GPU adds
significant overhead.


.. _guvectorize:

The ``@guvectorize`` decorator
==============================

While :func:`~numba.vectorize` allows you to write ufuncs that work on one
element at a time, the :func:`~numba.guvectorize` decorator takes the concept
one step further and allows you to write ufuncs that will work on an
arbitrary number of elements of input arrays, and take and return arrays of
differing dimensions.  The typical example is a running median or a
convolution filter.

Contrary to :func:`~numba.vectorize` functions, :func:`~numba.guvectorize`
functions don't return their result value: they take it as an array
argument, which must be filled in by the function.  This is because the
array is actually allocated by NumPy's dispatch mechanism, which calls into
the Numba-generated code.

Similar to :func:`~numba.vectorize` decorator, :func:`~numba.guvectorize`
also has two modes of operation: Eager, or decoration-time compilation and
lazy, or call-time compilation.


Here is a very simple example::

   @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
   def g(x, y, res):
       for i in range(x.shape[0]):
           res[i] = x[i] + y

The underlying Python function simply adds a given scalar (``y``) to all
elements of a 1-dimension array.  What's more interesting is the declaration.
There are two things there:

* the declaration of input and output *layouts*, in symbolic form:
  ``(n),()->(n)`` tells NumPy that the function takes a *n*-element one-dimension
  array, a scalar (symbolically denoted by the empty tuple ``()``) and
  returns a *n*-element one-dimension array;

* the list of supported concrete *signatures* as per ``@vectorize``; here,
  as in the above example, we demonstrate ``int64`` arrays.

.. note::
   1D array type can also receive scalar arguments (those with shape ``()``).
   In the above example, the second argument also could be declared as
   ``int64[:]``.  In that case, the value must be read by ``y[0]``.

We can now check what the compiled ufunc does, over a simple example::

   >>> a = np.arange(5)
   >>> a
   array([0, 1, 2, 3, 4])
   >>> g(a, 2)
   array([2, 3, 4, 5, 6])

The nice thing is that NumPy will automatically dispatch over more
complicated inputs, depending on their shapes::

   >>> a = np.arange(6).reshape(2, 3)
   >>> a
   array([[0, 1, 2],
          [3, 4, 5]])
   >>> g(a, 10)
   array([[10, 11, 12],
          [13, 14, 15]])
   >>> g(a, np.array([10, 20]))
   array([[10, 11, 12],
          [23, 24, 25]])


.. note::
   Both :func:`~numba.vectorize` and :func:`~numba.guvectorize` support
   passing ``nopython=True`` :ref:`as in the @jit decorator <jit-nopython>`.
   Use it to ensure the generated code does not fallback to
   :term:`object mode`.

.. _overwriting-input-values:

Overwriting input values
------------------------

In most cases, writing to inputs may also appear to work - however, this
behaviour cannot be relied on. Consider the following example function::

   @guvectorize([(float64[:], float64[:])], '()->()')
   def init_values(invals, outvals):
       invals[0] = 6.5
       outvals[0] = 4.2

Calling the `init_values` function with an array of `float64` type results in
visible changes to the input::

   >>> invals = np.zeros(shape=(3, 3), dtype=np.float64)
   >>> outvals = init_values(invals)
   >>> invals
   array([[6.5, 6.5, 6.5],
          [6.5, 6.5, 6.5],
          [6.5, 6.5, 6.5]])
   >>> outvals
   array([[4.2, 4.2, 4.2],
       [4.2, 4.2, 4.2],
       [4.2, 4.2, 4.2]])

This works because NumPy can pass the input data directly into the `init_values`
function as the data `dtype` matches that of the declared argument.  However, it
may also create and pass in a temporary array, in which case changes to the
input are lost. For example, this can occur when casting is required. To
demonstrate, we can  use an array of `float32` with the `init_values` function::

   >>> invals = np.zeros(shape=(3, 3), dtype=np.float32)
   >>> outvals = init_values(invals)
   >>> invals
   array([[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]], dtype=float32)

In this case, there is no change to the `invals` array because the temporary
casted array was mutated instead.

.. _dynamic-universal-functions:

Dynamic universal functions
===========================

As described above, if you do not pass any signatures to the
:func:`~numba.vectorize` decorator, your Python function will be used
to build a dynamic universal function, or :class:`~numba.DUFunc`.  For
example::

   from numba import vectorize

   @vectorize
   def f(x, y):
       return x * y

The resulting :func:`f` is a :class:`~numba.DUFunc` instance that
starts with no supported input types.  As you make calls to :func:`f`,
Numba generates new kernels whenever you pass a previously unsupported
input type.  Given the example above, the following set of interpreter
interactions illustrate how dynamic compilation works::

   >>> f
   <numba._DUFunc 'f'>
   >>> f.ufunc
   <ufunc 'f'>
   >>> f.ufunc.types
   []

The example above shows that :class:`~numba.DUFunc` instances are not
ufuncs.  Rather than subclass ufunc's, :class:`~numba.DUFunc`
instances work by keeping a :attr:`~numba.DUFunc.ufunc` member, and
then delegating ufunc property reads and method calls to this member
(also known as type aggregation).  When we look at the initial types
supported by the ufunc, we can verify there are none.

Let's try to make a call to :func:`f`::

   >>> f(3,4)
   12
   >>> f.types   # shorthand for f.ufunc.types
   ['ll->l']

If this was a normal Numpy ufunc, we would have seen an exception
complaining that the ufunc couldn't handle the input types.  When we
call :func:`f` with integer arguments, not only do we receive an
answer, but we can verify that Numba created a loop supporting C
:code:`long` integers.

We can add additional loops by calling :func:`f` with different inputs::

   >>> f(1.,2.)
   2.0
   >>> f.types
   ['ll->l', 'dd->d']

We can now verify that Numba added a second loop for dealing with
floating-point inputs, :code:`"dd->d"`.

If we mix input types to :func:`f`, we can verify that `Numpy ufunc
casting rules`_ are still in effect::

   >>> f(1,2.)
   2.0
   >>> f.types
   ['ll->l', 'dd->d']

.. _`Numpy ufunc casting rules`: http://docs.scipy.org/doc/numpy/reference/ufuncs.html#casting-rules

This example demonstrates that calling :func:`f` with mixed types
caused Numpy to select the floating-point loop, and cast the integer
argument to a floating-point value.  Thus, Numba did not create a
special :code:`"dl->d"` kernel.

This :class:`~numba.DUFunc` behavior leads us to a point similar to
the warning given above in "`The @vectorize decorator`_" subsection,
but instead of signature declaration order in the decorator, call
order matters.  If we had passed in floating-point arguments first,
any calls with integer arguments would be cast to double-precision
floating-point values.  For example::

   >>> @vectorize
   ... def g(a, b): return a / b
   ...
   >>> g(2.,3.)
   0.66666666666666663
   >>> g(2,3)
   0.66666666666666663
   >>> g.types
   ['dd->d']

If you require precise support for various type signatures, you should
specify them in the :func:`~numba.vectorize` decorator, and not rely
on dynamic compilation.

Dynamic generalized universal functions
=======================================

Similar to a dynamic universal function, if you do not specify any types to
the :func:`~numba.guvectorize` decorator, your Python function will be used
to build a dynamic generalized universal function, or :class:`~numba.GUFunc`.
For example::

   from numba import guvectorize

   @guvectorize('(n),()->(n)')
   def g(x, y, res):
       for i in range(x.shape[0]):
           res[i] = x[i] + y

We can verify the resulting function :func:`g` is a :class:`~numba.GUFunc`
instance that starts with no supported input types. For instance::

   >>> g
   <numba._GUFunc 'g'>
   >>> g.ufunc
   <ufunc 'g'>
   >>> g.ufunc.types
   []

Similar to a :class:`~numba.DUFunc`, as one make calls to :func:`g()`,
numba generates new kernels for previously unsupported input types. The
following set of interpreter interactions will illustrate how dynamic
compilation works for a :class:`~numba.GUFunc`::

   >>> x = np.arange(5, dtype=np.int64)
   >>> y = 10
   >>> res = np.zeros_like(x)
   >>> g(x, y, res)
   >>> res
   array([5, 6, 7, 8, 9])
   >>> g.types
   ['ll->l']

If this was a normal :func:`guvectorize` function, we would have seen an
exception complaining that the ufunc could not handle the given input types.
When we call :func:`g()` with the input arguments, numba creates a new loop
for the input types.

We can add additional loops by calling :func:`g` with new arguments::

   >>> x = np.arange(5, dtype=np.double)
   >>> y = 2.2
   >>> res = np.zeros_like(x)
   >>> g(x, y, res)

We can now verify that Numba added a second loop for dealing with
floating-point inputs, :code:`"dd->d"`.

   >>> g.types  # shorthand for g.ufunc.types
   ['ll->l', 'dd->d']

One can also verify that Numpy ufunc casting rules are working as expected::

   >>> x = np.arange(5, dtype=np.int64)
   >>> y = 2.2
   >>> res = np.zeros_like(x)
   >>> g(x, y, res)
   >>> res

If you need precise support for various type signatures, you should not rely on dynamic
compilation and instead, specify the types them as first
argument in the :func:`~numba.guvectorize` decorator.
