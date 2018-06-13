
.. _high-level-extending:

High-level extension API
========================

This extension API is exposed through the :mod:`numba.extending` module.


Implementing functions
----------------------

The ``@overload`` decorator allows you to implement arbitrary functions
for use in :term:`nopython mode` functions.  The function decorated with
``@overload`` is called at compile-time with the *types* of the function's
runtime arguments.  It should return a callable representing the
*implementation* of the function for the given types.  The returned
implementation is compiled by Numba as if it were a normal function
decorated with ``@jit``.  Additional options to ``@jit`` can be passed as
dictionary using the ``jit_options`` argument.

For example, let's pretend Numba doesn't support the :func:`len` function
on tuples yet.  Here is how to implement it using ``@overload``::

   from numba import types
   from numba.extending import overload

   @overload(len)
   def tuple_len(seq):
      if isinstance(seq, types.BaseTuple):
          n = len(seq)
          def len_impl(seq):
              return n
          return len_impl


You might wonder, what happens if :func:`len()` is called with something
else than a tuple? If a function decorated with ``@overload`` doesn't
return anything (i.e. returns None), other definitions are tried until
one succeeds.  Therefore, multiple libraries may overload :func:`len()`
for different types without conflicting with each other.

Implementing methods
--------------------

The ``@overload_method`` decorator similarly allows implementing a
method on a type well-known to Numba. The following example implements
the :meth:`~numpy.ndarray.take()` method on Numpy arrays::

   @overload_method(types.Array, 'take')
   def array_take(arr, indices):
      if isinstance(indices, types.Array):
          def take_impl(arr, indices):
              n = indices.shape[0]
              res = np.empty(n, arr.dtype)
              for i in range(n):
                  res[i] = arr[indices[i]]
              return res
          return take_impl

Implementing attributes
-----------------------

The ``@overload_attribute`` decorator allows implementing a data
attribute (or property) on a type.  Only reading the attribute is
possible; writable attributes are only supported through the
:ref:`low-level API <low-level-extending>`.

The following example implements the :attr:`~numpy.ndarray.nbytes` attribute
on Numpy arrays::

   @overload_attribute(types.Array, 'nbytes')
   def array_nbytes(arr):
      def get(arr):
          return arr.size * arr.itemsize
      return get

Importing Cython Functions
--------------------------

The function ``get_cython_function_address`` obtains the address of a
C function in a Cython extension module. The address can be used to
access the C function via a :func:`ctypes.CFUNCTYPE` callback, thus
allowing use of the C function inside a Numba jitted function. For
example, suppose that you have the file ``foo.pyx``::

   from libc.math cimport exp

   cdef api double myexp(double x):
       return exp(x)

You can access ``myexp`` from Numba in the following way::

   import ctypes
   from numba.extending import get_cython_function_address

   addr = get_cython_function_address("foo", "myexp")
   functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
   myexp = functype(addr)

The function ``myexp`` can now be used inside jitted functions, for
example::

   @njit
   def double_myexp(x):
       return 2*myexp(x)

One caveat is that if your function uses Cython's fused types, then
the function's name will be mangled. To find out the mangled name of
your function you can check the extension module's ``__pyx_capi__``
attribute.
