
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
decorated with ``@jit``.

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

Finally, the ``@overload_attribute`` decorator allows implementing a data
attribute (or property) on a type.  Only reading the attribute is possible;
writable attributes are only supported through the
:ref:`low-level API <low-level-extending>`.

The following example implements the :attr:`~numpy.ndarray.nbytes` attribute
on Numpy arrays::

   @overload_attribute(types.Array, 'nbytes')
   def array_nbytes(arr):
      def get(arr):
          return arr.size * arr.itemsize
      return get

