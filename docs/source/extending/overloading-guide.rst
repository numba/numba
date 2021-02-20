
.. _overloading-guide:

==============================
A guide to using ``@overload``
==============================


As mentioned in the :ref:`high-level extension API <high-level-extending>`, you
can use the ``@overload`` decorator to create a Numba implementation of a
function that can be used in :term:`nopython mode` functions. A common use case
is to re-implement NumPy functions so that they can be called in ``@jit``
decorated code. This section discusses how and when to use the ``@overload``
decorator and what contributing such a function to the Numba code base might
entail. This should help you get started when needing to use the ``@overload``
decorator or when attempting to contribute new functions to Numba itself.

The ``@overload`` decorator and it's variants are useful when you have a
third-party library that you do not control and you wish to provide Numba
compatible implementations for specific functions from that library.

Concrete Example
================

Let's assume that you are working on a minimization algorithm that makes use of
|scipy.linalg.norm|_ to find different vector norms and the `frobenius
norm <https://en.wikipedia.org/wiki/Frobenius_inner_product>`_ for matrices.
You know that only integer and real numbers will be involved. (While this may
sound like an artificial example, especially because a Numba implementation of
``numpy.linalg.norm`` exists, it is largely pedagogical and serves to
illustrate how and when to use ``@overload``).

.. |scipy.linalg.norm| replace:: ``scipy.linalg.norm``
.. _scipy.linalg.norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html

The skeleton might look something like this::

    def algorithm():
        # setup
        v = ...
        while True:
            # take a step
            d = scipy.linalg.norm(v)
            if d < tolerance:
                break

Now, let's further assume, that you have heard of Numba and you now wish to use
it to accelerate your function. However, after adding the
``jit(nopython=True)``
decorator, Numba complains that ``scipy.linalg.norm`` isn't supported. From
looking at the documentation, you realize that a norm is probably fairly easy
to implement using NumPy. A good starting point is the following template.

.. literalinclude:: template.py

After some deliberation and tinkering, you end up with the following code:

.. literalinclude:: mynorm.py

As you can see, the implementation only supports what you need right now:

* Only supports integer and floating-point types
* All vector norms
* Only the Frobenius norm for matrices
* Code sharing between vector and matrix implementations using
  ``@register_jitable``.
* Norms are implemented using NumPy syntax. (This is possible because
  Numba is very aware of NumPy and many functions are supported.)

So what actually happens here? The ``overload`` decorator registers a suitable
implementation for ``scipy.linalg.norm`` in case a call to this is encountered
in code that is being JIT-compiled, for example when you decorate your
``algorithm`` function with ``@jit(nopython=True)``. In that case, the function
``jit_norm`` will be called with the currently encountered types and will then
return either ``_oneD_norm_x`` in the vector case and ``_two_D_norm_2``.

You can download the example code here: :download:`mynorm.py </extending/mynorm.py>`

Implementing ``@overload`` for NumPy functions
==============================================

Numba supports NumPy through the provision of ``@jit`` compatible
re-implementations of NumPy functions. In such cases ``@overload`` is a very
convenient option for writing such implementations, however there are a few
additional things to watch out for.

* The Numba implementation should match the NumPy implementation as closely as
  feasible with respect to accepted types, arguments, raised exceptions and
  algorithmic complexity (Big-O / Landau order).

* When implementing supported argument types, bear in mind that, due to
  duck typing, NumPy does tend to accept a multitude of argument types beyond
  NumPy arrays such as scalar, list, tuple, set, iterator, generator etc.
  You will need to account for that during type inference and subsequently as
  part of the tests.

* A NumPy function may return a scalar, array or a data structure
  which matches one of its inputs, you need to be aware of type
  unification problems and dispatch to appropriate implementations. For
  example, |np.corrcoef|_ may return an array or a scalar depending on its
  inputs.

.. |np.corrcoef| replace:: ``np.corrcoef``
.. _np.corrcoef: https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

* If you are implementing a new function, you should always update the
  `documentation
  <https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html>`_.
  The sources can be found in ``docs/source/reference/numpysupported.rst``. Be
  sure to mention any limitations that your implementation has, e.g. no support
  for the ``axis`` keyword.

* When writing tests for the functionality itself, it's useful to include
  handling of non-finite values, arrays with different shapes and layouts,
  complex inputs, scalar inputs, inputs with types for which support is not
  documented (e.g. a function which the NumPy docs say requires a float or int
  input might also 'work' if given a bool or complex input).

* When writing tests for exceptions, for example if adding tests to
  ``numba/tests/test_np_functions.py``, you may encounter the following error
  message:

  .. code::

        ======================================================================
        FAIL: test_foo (numba.tests.test_np_functions.TestNPFunctions)
        ----------------------------------------------------------------------
        Traceback (most recent call last):
        File "<path>/numba/numba/tests/support.py", line 645, in tearDown
            self.memory_leak_teardown()
        File "<path>/numba/numba/tests/support.py", line 619, in memory_leak_teardown
            self.assert_no_memory_leak()
        File "<path>/numba/numba/tests/support.py", line 628, in assert_no_memory_leak
            self.assertEqual(total_alloc, total_free)
        AssertionError: 36 != 35

  This occurs because raising exceptions from jitted code leads to reference
  leaks. Ideally, you will place all exception testing in a separate test
  method and then add a call in each test to ``self.disable_leak_check()`` to
  disable the leak-check (inherit from ``numba.tests.support.TestCase`` to make
  that available).

* For many of the functions that are available in NumPy, there are
  corresponding methods defined on the NumPy ``ndarray`` type. For example, the
  function ``repeat`` is available as a NumPy module level function and a
  member function on the ``ndarray`` class.

  .. code:: python

        import numpy as np
        a = np.arange(10)
        # function
        np.repeat(a, 10)
        # method
        a.repeat(10)

  Once you have written the function implementation, you can easily use
  ``@overload_method`` and reuse it. Just be sure to check that NumPy doesn't
  diverge in the implementations of its function/method.

  As an example, the ``repeat`` function/method:

  .. code:: python

        @extending.overload_method(types.Array, 'repeat')
        def array_repeat(a, repeats):
            def array_repeat_impl(a, repeat):
                # np.repeat has already been overloaded
                return np.repeat(a, repeat)

            return array_repeat_impl

* If you need to create ancillary functions, for example to re-use a small
  utility function or to split your implementation across functions for the
  sake of readability, you can make use of the ``@register_jitable`` decorator.
  This will make those functions available from within your ``@jit`` and
  ``@overload`` decorated functions.

* The Numba continuous integration (CI) set up tests a wide variety of NumPy
  versions, you'll sometimes be alerted to a change in behaviour from some
  previous NumPy version. If you can find supporting evidence in the NumPy
  change log / repository, then you'll need to decide whether to create
  branches and attempt to replicate the logic across versions, or use a version
  gate (with associated wording in the documentation) to advertise that Numba
  replicates NumPy from some particular version onwards.

* You can look at the Numba source code for inspiration, many of the overloaded
  NumPy functions and methods are in ``numba/targets/arrayobj.py``. Below, you
  will find a list of implementations to look at that are well implemented in
  terms of accepted types and test coverage.

  * ``np.repeat``
