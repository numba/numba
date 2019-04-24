
.. _overloading-guide:

============================
A guide to using ``@overload``
============================


As mentioned in the :ref:`high-level extension API <high-level-extending>`, you
can use the ``@overload`` decorator to create a Numba implementation of a function
that can be used in :term:`nopython mode` functions. A common use case
is to re-implement NumPy functions so that they can be called in ``jit``
decorated code. In this section will discuss what contributing such a function
to Numba might entail. This should help you get started when attempting to
contribute new overloaded functions to Numba.

Annotated Template
==================

Here is an annotated template that outlines how the specific parts ought to
looks like. This should give you an idea as to the structure required.

.. literalinclude:: template.py


Concrete Example
================

Let's assume that you have a module called ``mymodule.py`` which implements a single
a function called ``set_to_x``:

.. literalinclude:: mymodule.py

Usually, you use this function to set all elements of a NumPy array to a
specific value. Now, you do some profiling and you realize that our function
might be a bit slow.

.. code:: pycon

    In [1]: import numpy as np

    In [2]: from mymodule import set_to_x

    In [3]: a = np.arange(100000)

    In [4]: %timeit set_to_x(a, 1)
    5.88 ms ± 43.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

Since this function is used very often in your ``jit`` decorated
functions, for example in ``myalgorithm``, you choose to use to ``overload`` in
an attempt to accelerate things.

.. code:: python

    @njit
    def myalgorithm(a, x):
        # algorithm code
        ham.set_to_x(a, x)
        # algorithm code

Inspired by the template above, your implementation might look something like:

.. literalinclude:: myjitmodule.py



When the function is timed, you find yourself pleasantly surprised:

.. code:: pycon

    In [1]: import numpy as np

    In [2]: import myjitmodule

    In [3]: a = np.arange(100000)

    In [4]: %timeit myjitmodule.myalgorithm(a, 1)
    17.6 µs ± 327 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

But of course, your implementation doesn't generalize to your colleague's
use-case, who would like to use ``set_to_x`` with a floating point number
instead:

.. code:: pycon

    In [4]: myjitmodule.myalgorithm(a, 1.0)
    TypingError: Failed in nopython mode pipeline (step: nopython frontend)
    Invalid use of Function(<function set_to_x at 0x11aea71e0>) with argument(s) of type(s): (array(int64, 1d, C), float64)
    * parameterized
    In definition 0:
        All templates rejected with literals.
    In definition 1:
        All templates rejected without literals.
    This error is usually caused by passing an argument of a type that is unsupported by the named function.
    [1] During: resolving callee type: Function(<function set_to_x at 0x11aea71e0>)
    [2] During: typing of call at /Users/vhaenel/git/numba/docs/source/extending/myjitmodule.py (25)


    File myjitmodule.py", line 25:
    def myalgorithm(a, x):
        # algorithm code
        mymodule.set_to_x(a, x)
        # algorithm code


Providing multiple implementations and dispatching based on types
=================================================================

As you saw above, the overload implementation for ``set_to_x`` function
doesn't accept floating-point arguments. Let's extended the specification of
the function as follows:

* The numerical type of the array ``arr`` must match the numerical type of the
  scalar ``x`` argument, i.e. if ``arr`` is of type ``int64``, then ``x`` must
  be of this type too.
* Only integer and floating-point types are to be supported for argument ``x``.
* No ``nan`` values are allowed in ``arr`` when it is of floating-point type and if
  such a value is encountered an appropriate ``ValueError`` should be raised.
* If a tuple is used instead of an array as a value for ``arr``, a custom error message with a hint
  for the user should be issued.

The resulting implementation could look as follows:

.. literalinclude:: myjitmodule2.py

As you can see, the typing checking code has been increased significantly to
match the new requirements. Also, multiple implementations---one for integers
and one for floating-point---are provided. We check inside the typing scope
which implementation should be used and also raise any custom error messages
required. Importantly, the check for ``nan`` values is only present in the
floating point implementation as this additional check creates a runtime
overhead. This can easily be observed during profiling:

.. code:: pycon

    In [1]: a = np.arange(100, dtype=np.float64)

    In [2]: %timeit myalgorithm(a, 1.0)
    473 ns ± 11.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

    In [3]: a = np.arange(100)

    In [4]: %timeit myalgorithm(a, 1)
    237 ns ± 4.59 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

Writing Tests
=============

The following is a sufficient piece of test code for the overloaded
``set_to_x`` implementation. As you can see, only a small part of the test-code
is about testing if the function works correctly. Most of the test code in this
example checks that all error cases are handled and that all raised exceptions
are of the correct type and have the correct error message. When implementing
tests cases for Numba, you should always use ``numba.tests.support.TestCase``.

.. literalinclude:: test_myjitmodule2.py

While it is a pretty decent test, it wouldn't be accepted into Numba.
There are a few more test-cases that should be implemented:

* Empty array
* Single value array
* Multidimensional arrays
* Tests for ``int32`` and ``float32`` types

You can download the example code from above including tests here:

* :download:`mymodule.py </extending/mymodule.py>`
* :download:`myjitmodule.py </extending/myjitmodule.py>`
* :download:`myjitmodule2.py </extending/myjitmodule2.py>`
* :download:`test_myjitmodule2.py </extending/test_myjitmodule2.py>`

Implementing ``@overload``s for NumPy functions
=============================================

Numba supports NumPy through the provision of ``jit`` compatible
re-implementations of NumPy functions. In such cases ``overload`` is a very
convenient option for writing such implementations, however there are a few additional things to watch out for.

* The Numba implementation should match the NumPy implementation as closely as
  feasible with respect to accepted types, arguments, raised exceptions and
  runtime (Big-O / Landau order).

* When implementing supported argument types, bear in mind that---thanks to
  duck typing---NumPy does tend to accept a multitude of argument types beyond
  NumPy arrays such as scalar, list, tuple, set, iterator, generator etc.. So
  you will need to account for that during type inference and subsequently as
  part of the tests.

* A NumPy function may return a scalar, array or a data structure
  which matches one of its inputs - so you need to watch out for type
  unification problems and dispatch to appropriate implementations. For
  example, ``np.corrcoef`` may return an array or a scalar depending on it's
  inputs.

* If you are implementing a new function, you should always update the
  `documentation
  <http://numba.pydata.org/numba-doc/latest/reference/numpysupported.html>`_.
  The sources can be found in `docs/source/reference/numpysupported.rst``. Be
  sure to mention any limitations that your implementation has, e.g. no support
  for the ``axis`` keyword.

* When writing tests for the functionality itself, it's useful to include
  handling of non-finite values; arrays with different shapes and layouts;
  complex inputs; scalar inputs; inputs with types for which support is not
  documented (e.g. a function which the NumPy docs say requires a float or int
  input might also 'work' if given a bool or complex input).

* When writing tests for exceptions, for example if adding tests to
  ``numba/tests/test_np_functions.py``, you may encounter the following error
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

  This occurs because raising exceptions from jitted code leads to reference leaks. Ideally, you will
  place all exception testing in a separate test method and then add a call in each test to
  ``self.disable_leak_check()`` to disable the leak-check. (Remember to inherit
  from ``numba.tests.support.TestCase`` to make that available).

* For many of the functions that are available in NumPy, there are
  corresponding methods defined on the NumPy array type. For example, the
  function ``repeat`` is available in two flavours.

  .. code:: python

        import numpy as np
        a = np.arange(10)
        # function
        np.repeat(a, 10)
        # method
        a.repeat(10)

  Once you have written the function implementation, you can easily use
  ``overload_method`` and reuse it. Just be sure to check that NumPy doesn't
  diverge in the implementations of its function/method.

  For example for the ``repeat`` function/method:

  .. code:: python

        @extending.overload_method(types.Array, 'repeat')
        def array_repeat(a, repeats):
            def array_repeat_impl(a, repeat):
                # np.repeat has already been overloaded
                return np.repeat(a, repeat)

            return array_repeat_impl

* If you need to create ancillary functions, for example to re-use a small
  utility function or to split your implementation across functions for the
  sake of readability, you can make use of the ``register_jitable`` decorator.
  This will make those functions available from within your ``jit`` and
  ``overload`` decorated functions.

* The Numba continuous integration (CI) setup  tests a wide variety of NumPy
  versions---you'll sometimes be alerted to a change in behaviour back in some
  previous NumPy version. If you can find supporting evidence in the NumPy
  change log / repo, then you'll need to decide whether to branch logic and
  attempt to replicate the logic across versions, or use a version gate (with
  associated wording in the docs) to advertise that Numba replicates NumPy from
some particular version onwards.

* You can look at the Numba source code for inspiration, many of the overloaded
  NumPy functions and methods are in ``numba/targets/arrayobj.py``. Below, you
  will find a list of implementations to look at, that are well impemented in
  terms of accepted types and test coverage.

  * ``np.repeat``
