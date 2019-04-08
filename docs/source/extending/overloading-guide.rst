
.. _overloading-guide:

============================
Guide to using ``@overload``
============================


As mentioned in the :ref:`high-level extension API <high-level-extending>`, you
can use the ``@overload`` decorator to provide a Numba specific implementation
that can be used in :term:`nopython mode` functions. A common used case of this
is to implement Numpy functions so that they can be called in jitted code. In
this section will discuss what contributing such a function to Numba might
entail. This should help you get started when attempting to contribute new
overloaded functions to Numba.

Annotated Template
==================

Here is an annotated template that outlines how the specific parts ought to
looks like. This should give you an idea as to the structure required.

.. code:: python

    # Declare that function `eggs` is going to be overloaded (have a
    # substitutable numba implementation)
    @overload(eggs)

    # Define the overload function with formal arguments
    # these arguments must be matched in the inner function implementation
    def jit_eggs(arg0, arg1, arg2, ...):

         # This scope is for typing, access is available to the *type* of all
         # arguments. This information can be used to change  the behaviour of the
         # implementing function and check that the types are  actually supported
         # by the implementation.

        print(arg0) # this will show the Numba type of arg0

        # This is the definition of the function that implements the `eggs` work. It
        # does whatever algorithm is needed to implement eggs.
        def eggs_impl(arg0, arg1, arg2, ...): # match arguments to jit_eggs above
            # < Implementation goes here >
            return # whatever needs to be returned by the algorithm

        # return the implementation
        return eggs_impl

Concrete Example
================

Let's assume that you have a module called ``ham.py`` which implements a single
a function  called ``set_to_x``:

.. code:: python

    def set_to_x(arr, x):
        for i in len(arr):
            arr[i] = x

Usually, you use this function to set all elements of a Numpy array to a
specific value. Now, you do some profiling and you realize, that our function
might be a bit slow.

.. code:: pycon

    In [1]: import numpy as np

    In [2]: from ham import set_to_x

    In [3]: a = np.arange(100000)

    In [4]: %timeit set_to_x(a, 1)
    5.88 ms ± 43.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

And since you use this function, very often in your jitted functions, for
example in ``breakfast``, you choose to use to ``overload`` in an attempt to
accelerate things.

.. code:: python

    @njit
    def breakfast(a, x):
        ham.set_to_x(a, x)

Inspired by the template above, your implementation might look something like:

.. code:: python

    # Numba imports
    from numba import njit, types
    from numba.extending import overload


    # Import the module, where you wish to overload something
    import ham


    # decorate with overload
    @overload(ham.set_to_x)
    def set_to_x_jit(arr, x):
        # This is the *typing scope*. `arr` and `x` are not the array and the
        # scalar itself but their types. So, you can implement type-checking
        # logic here. In this case, we check that `arr` is a Numpy array and that
        # `x` is an integer

        if not isinstance(arr, types.Array):
            return
        if not isinstance(x, types.Integer):
            return

        # This is the *optimized* implementation
        def set_to_x_impl(arr, x):
            arr[:] = x

        # Return this implementation itself
        return set_to_x_impl

    @njit
    def breakfast(a, x):
        ham.set_to_x(a, x)

And when you go to profile it, you find yourself pleasantly surprised:

.. code:: pycon

    In [1]: import numpy as np

    In [2]: import spam

    In [3]: a = np.arange(100000)

    In [4]: %timeit spam.breakfast(a, 1)
    17.6 µs ± 327 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

But of course, your implementation doesn't generalize to your colleagues
use-case, who would like to use ``set_to_x`` with a floating point number
instead:

.. code:: pycon

    In [4]: spam.breakfast(a, 1.0)
    TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    Invalid use of Function(<function set_to_x at 0x11aea71e0>) with argument(s) of type(s): (array(int64, 1d, C), float64)
    * parameterized
    In definition 0:
        All templates rejected with literals.
    In definition 1:
        All templates rejected without literals.
    This error is usually caused by passing an argument of a type that is unsupported by the named function.
    [1] During: resolving callee type: Function(<function set_to_x at 0x11aea71e0>)
    [2] During: typing of call at /Users/vhaenel/git/numba/spam.py (25)


    File "spam.py", line 25:
    def breakfast(a, x):
        ham.set_to_x(a, x)


Providing multiple implementations and dispatching based on types
=================================================================

As you saw above, the overload ``set_to_x`` function doesn't accept floating
point arguments. Let's extended the specification of the function as follows:

* The numerical type of the array ``arr`` must match the numerical type of the
  scalar ``x`` argument
* Only integer and floating-point types are to be supported
* For the floating-point type, no ``nan`` values are allowed in ``arr`` and if
  such a value is encountered, an appropriate ``ValueError`` should be raised.
* If a tuple is used instead of an array, a custom error message with a hint
  for the user should be issued.

The resulting implementation could look as follows:

.. code:: python


    @overload(ham.set_to_x)
    def set_to_x_jit_v2(arr, x):

        # implementation for integers
        def set_to_x_impl_int(arr, x):
            arr[:] = x

        # implementation for floating-point
        def set_to_x_impl_float(arr, x):
            if np.any(np.isnan(arr)):
                raise ValueError("no element of arr must be nan")
            arr[:] = x

        # check that it is an array
        if isinstance(arr, types.Array):
            # validate that arr and x have the same type
            if arr.dtype == x:
                if isinstance(x, types.Integer):
                    # dispatch for integers
                    return set_to_x_impl_int
                elif isinstance(x, types.Float):
                    # dispatch for float
                    return set_to_x_impl_float
                else:
                    # must be some other type
                    raise TypingError(
                        "only integer and floating-point types allowed")
            else:
                # type mismatch
                raise TypingError("the types of the input do not match")
        elif isinstance(arr, types.BaseTuple):
            # custom error for tuple as input
            raise TypingError("tuple isn't allowed as input, use numpy arrays")

        # fall through, None returned as no suitable implementation was found

As you can see, the typing checking code has been increased significantly to
match the new requirements. Also, multiple implementations---one for integers
and one for floating-point---are provided. We check inside the typing scope
which implementation should be used and also raise any custom error messages
required. Importantly, the check for `nan` values is only present in the
floating point implementation as this additional check creates a runtime
overhead. This can easily be observed during profiling:

.. code:: pycon

    In [1]: a = np.arange(100, dtype=np.float64)

    In [2]: %timeit breakfast(a, 1.0)
    473 ns ± 11.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    In [3]: a = np.arange(100)

    In [4]: a = np.arange(100)
    237 ns ± 4.59 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

Writing Tests
=============

The following is a sufficient piece of test code for the overloaded
``set_to_x`` implementation. As you can see, only a small part of the test-code
is about testing if the function works correctly. Most of the test code in this
example checks that all error cases are handled and that all raised exceptions
are of the correct type and have the correct error message. When implementing
tests cases for Numba, you should always use ``numba.tests.support.TestCase``.

.. code:: python

    import numpy as np
    from numba import njit
    from numba import unittest_support as unittest
    from numba.tests import support
    from numba.errors import TypingError

    import ham
    import spam # noqa - has side-effect, overload ham.set_to_x


    @njit
    def wrap_set_to_x(arr, x):
        ham.set_to_x(arr, x)


    class TestSpam(support.TestCase):

        def test_int(self):
            a = np.arange(10)
            wrap_set_to_x(a, 1)
            self.assertPreciseEqual(np.ones(10, dtype=np.int64), a)

        def test_float(self):
            a = np.arange(10, dtype=np.float64)
            wrap_set_to_x(a, 1.0)
            self.assertPreciseEqual(np.ones(10), a)

        def test_float_exception_on_nan(self):
            a = np.arange(10, dtype=np.float64)
            a[0] = np.nan
            with self.assertRaises(ValueError) as e:
                wrap_set_to_x(a, 1.0)
            self.assertIn("no element of arr must be nan",
                        str(e.exception))

        def test_type_mismatch(self):
            a = np.arange(10)
            with self.assertRaises(TypingError) as e:
                wrap_set_to_x(a, 1.0)
            self.assertIn("the types of the input do not match",
                        str(e.exception))

        def test_exception_on_unsupported_dtype(self):
            a = np.arange(10, dtype=np.complex128)
            with self.assertRaises(TypingError) as e:
                wrap_set_to_x(a, np.complex128(1.0))
            self.assertIn("only integer and floating-point types allowed",
                        str(e.exception))

        def test_exception_on_tuple(self):
            a = (1, 2, 3)
            with self.assertRaises(TypingError) as e:
                wrap_set_to_x(a, 1)
            self.assertIn("tuple isn't allowed as input, use numpy arrays",
                        str(e.exception))


    if __name__ == '__main__':
        unittest.main()

While is is a pretty descent test, it wouldn't be accepted into the Numba.
There are a few more test-cases that should be implemented:

* Empty array
* Single value array
* Multidimensional arrays
* Tests for ``int32`` and ``float32`` types
* Exclude 64 bit tests on 32 bit machines

Implementing ``overload`` for Numpy functions
=============================================

When contributing Numpy ``overload`` s to Numba, there are a few additional
things to watch out for.

* The  Numba implementation should match the Numpy implementation as closely as
  feasible with respect to accepted types, arguments, raised exceptions and
  runtime (Big-O / Landau order).

* If you are implementing a new function, you should always update the
  `documentation
  <http://numba.pydata.org/numba-doc/latest/reference/numpysupported.html>`_.
  The sources can be found in `docs/source/reference/numpysupported.rst``. Be
  sure to mention any limitations that your implementation has.

* When writing tests for exceptions, for example, when adding tests to
  ``numba/tests/test_np_functions.py`` you may encounter the following error
  message:

  .. code:: pycon

        ======================================================================
        FAIL: test_foo (numba.tests.test_np_functions.TestNPFunctions)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
        File "/Users/vhaenel/git/numba/numba/tests/support.py", line 645, in tearDown
            self.memory_leak_teardown()
        File "/Users/vhaenel/git/numba/numba/tests/support.py", line 619, in memory_leak_teardown
            self.assert_no_memory_leak()
        File "/Users/vhaenel/git/numba/numba/tests/support.py", line 628, in assert_no_memory_leak
            self.assertEqual(total_alloc, total_free)
        AssertionError: 36 != 35

  This is caused because some exceptions leak references. Ideally, you will
  place all exception testing in a separate test method and then add a call to 
  ``self.disable_leak_check()`` to disable the leak-check.

* For many of the functions that are available in Numpy, there are
  corresponding methods defined on the numpy array type. For example, the
  function ``repeat`` is available in two flavours.

  .. code:: python

        import numpy as np
        a = np.arange(10)
        # function
        np.repeat(a, 10)
        # method
        a.repeat(10)

  Once you have written the function implementation, you can easily use
  ``overload_method`` and reuse it, for example for the ``repeat`` function/method.
  Just be sure to check that Numpy doesn't diverge in the implementations of
  it's function/method.

  .. code:: python

        @extending.overload_method(types.Array, 'repeat')
        def array_repeat(a, repeats):
            def array_repeat_impl(a, repeat):
                return np.repeat(a, repeat)

            return array_repeat_impl

* If you need to create ancillary functions, for example to re-use a small
  utility function or to split your implementation across functions for the
  sake of readability, you can make use of the ``register_jitable`` decorator.
  This will make those functions available from within your jitted functions.

* You can look at the Numba source code for inspiration, much of the overloaded
  Numpy functions and methods are in ``numba/targets/arrayobj.py``. Good
  implementations to look at are:

  * ``np.repeat``
