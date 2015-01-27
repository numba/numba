==================================
Creating Numpy universal functions
==================================


The ``@vectorize`` decorator
============================

Numba's vectorize allows Python functions taking scalar input arguments to
be used as NumPy `ufuncs`_.  Creating a traditional NumPy ufunc is not
not the most straightforward process and involves writing some C code.
Numba makes this easy.  Using the :func:`~numba.vectorize` decorator, Numba
can compile a pure Python function into a ufunc that operates over NumPy
arrays as fast as traditional ufuncs written in C.

.. _ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html

Using :func:`~numba.vectorize`, you write your function as operating over
input scalars, rather than arrays.  Numba will generate the surrounding
loop (or *kernel*) allowing efficient iteration over the actual inputs.

The :func:`~numba.vectorize` decorator needs you to pass a list of signatures
you want to support.  In the basic case, only one signature will be passed::

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


The ``@guvectorize`` decorator
==============================

While :func:`~numba.vectorize` allows you to write ufuncs that work on one
element at a time, the :func:`~numba.guvectorize` decorator takes the concept
one step further and allows you to write ufuncs that will work on an
arbitrary number of elements of input arrays, and take and return arrays of
differing dimensions.  The typical example is a running median or a
convolution filter.

Contrary to :func:`~numba.vectorize` functions, :func:`~numba.guvectorize`
functions don't return their result value: their take it as an array
argument, which must be filled in by the function.  This is because the
array is actually allocated by NumPy's dispatch mechanism, which calls into
the Numba-generated code.

Here is a very simple example::

   @guvectorize([(int64[:], int64[:], int64[:])], '(n),()->(n)')
   def g(x, y, res):
       for i in range(x.shape[0]):
           res[i] = x[i] + y[0]

The underlying Python function simply adds a given scalar (``y``) to all
elements of a 1-dimension array.  What's more interesting is the declaration.
There are two things there:

* the declaration of input and output *layouts*, in symbolic form:
  ``(n),()->(n)`` tells NumPy that the function takes a *n*-element one-dimension
  array, a scalar (symbolically denoted by the empty tuple ``()``) and
  returns a *n*-element one-dimension array;

* the list of supported concrete *signatures* as in ``@vectorize``; here we
  only support ``int64`` arrays.

.. note::
   The concrete signature does not allow for scalar values, even though
   the layout may mention them.  In this example, the second argument is
   declared as ``int64[:]``, not ``int64``.
   This is why it must be dereferenced by fetching ``y[0]``.

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
