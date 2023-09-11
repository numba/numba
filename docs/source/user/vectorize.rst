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
  signatures to the decorator, you will be building a NumPy universal
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
into a NumPy ufunc.  In the basic case, only one signature will be
passed:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_one_signature`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_one_signature.begin
   :end-before: magictoken.ex_vectorize_one_signature.end
   :dedent: 12
   :linenos:

If you pass several signatures, beware that you have to pass most specific
signatures before least specific ones (e.g., single-precision floats
before double-precision floats), otherwise type-based dispatching will not work
as expected:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_multiple_signatures`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_multiple_signatures.begin
   :end-before: magictoken.ex_vectorize_multiple_signatures.end
   :dedent: 12
   :linenos:

The function will work as expected over the specified array types:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_multiple_signatures`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_return_call_one.begin
   :end-before: magictoken.ex_vectorize_return_call_one.end
   :dedent: 12
   :linenos:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_multiple_signatures`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_return_call_two.begin
   :end-before: magictoken.ex_vectorize_return_call_two.end
   :dedent: 12
   :linenos:

but it will fail working on other types::

   >>> a = np.linspace(0, 1+1j, 6)
   >>> f(a, a)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: ufunc 'ufunc' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''


You might ask yourself, "why would I go through this instead of compiling
a simple iteration loop using the :ref:`@jit <jit>` decorator?".  The
answer is that NumPy ufuncs automatically get other features such as
reduction, accumulation or broadcasting.  Using the example above:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_multiple_signatures`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_return_call_three.begin
   :end-before: magictoken.ex_vectorize_return_call_three.end
   :dedent: 12
   :linenos:

.. seealso::
   `Standard features of ufuncs <http://docs.scipy.org/doc/numpy/reference/ufuncs.html#ufunc>`_ (NumPy documentation).

.. note::
   Only the broadcasting and reduce features of ufuncs are supported in compiled code.

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


Starting in Numba 0.59, the ``cpu`` target supports the following attributes
and methods in compiled code:

- ``ufunc.nin``
- ``ufunc.nout``
- ``ufunc.nargs``
- ``ufunc.identity``
- ``ufunc.signature``
- ``ufunc.reduce()`` (only the first 5 arguments - experimental feature)

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


Here is a very simple example:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize.begin
   :end-before: magictoken.ex_guvectorize.end
   :dedent: 12
   :linenos:

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

We can now check what the compiled ufunc does, over a simple example:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_call_one.begin
   :end-before: magictoken.ex_guvectorize_call_one.end
   :dedent: 12
   :linenos:

The nice thing is that NumPy will automatically dispatch over more
complicated inputs, depending on their shapes:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_call_two.begin
   :end-before: magictoken.ex_guvectorize_call_two.end
   :dedent: 12
   :linenos:


.. note::
   Both :func:`~numba.vectorize` and :func:`~numba.guvectorize` support
   passing ``nopython=True`` :ref:`as in the @jit decorator <jit-nopython>`.
   Use it to ensure the generated code does not fallback to
   :term:`object mode`.


.. _scalar-return-values:

Scalar return values
--------------------

Now suppose we want to return a scalar value from 
:func:`~numba.guvectorize`. To do this, we need to:

* in the signatures, declare the scalar return with ``[:]`` like 
  a 1-dimensional array (eg. ``int64[:]``),

* in the layout, declare it as ``()``,

* in the implementation, write to the first element (e.g. ``res[0] = acc``).

The following example function computes the sum of the 1-dimensional 
array (``x``) plus the scalar (``y``) and returns it as a scalar:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_scalar_return`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_scalar_return.begin
   :end-before: magictoken.ex_guvectorize_scalar_return.end
   :dedent: 12
   :linenos:

Now if we apply the wrapped function over the array, we get a scalar 
value as the output:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_scalar_return`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_scalar_return_call.begin
   :end-before: magictoken.ex_guvectorize_scalar_return_call.end
   :dedent: 12
   :linenos:


.. _overwriting-input-values:

Overwriting input values
------------------------

In most cases, writing to inputs may also appear to work - however, this
behaviour cannot be relied on. Consider the following example function:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_overwrite`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_overwrite.begin
   :end-before: magictoken.ex_guvectorize_overwrite.end
   :dedent: 12
   :linenos:

Calling the `init_values` function with an array of `float64` type results in
visible changes to the input:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_overwrite`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_overwrite_call_one.begin
   :end-before: magictoken.ex_guvectorize_overwrite_call_one.end
   :dedent: 12
   :linenos:

This works because NumPy can pass the input data directly into the `init_values`
function as the data `dtype` matches that of the declared argument.  However, it
may also create and pass in a temporary array, in which case changes to the
input are lost. For example, this can occur when casting is required. To
demonstrate, we can  use an array of `float32` with the `init_values` function:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_overwrite`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_overwrite_call_two.begin
   :end-before: magictoken.ex_guvectorize_overwrite_call_two.end
   :dedent: 12
   :linenos:

In this case, there is no change to the `invals` array because the temporary
casted array was mutated instead.

To solve this problem, one needs to tell the GUFunc engine that the ``invals``
argument is writable. This can be achieved by passing ``writable_args=('invals',)``
(specifying by name), or ``writable_args=(0,)`` (specifying by position) to
``@guvectorize``. Now, the code above works as expected:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_overwrite`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_overwrite_call_three.begin
   :end-before: magictoken.ex_guvectorize_overwrite_call_three.end
   :dedent: 12
   :linenos:

.. _dynamic-universal-functions:

Dynamic universal functions
===========================

As described above, if you do not pass any signatures to the
:func:`~numba.vectorize` decorator, your Python function will be used
to build a dynamic universal function, or :class:`~numba.DUFunc`.  For
example:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_dynamic.begin
   :end-before: magictoken.ex_vectorize_dynamic.end
   :dedent: 12
   :linenos:

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

Let's try to make a call to :func:`f`:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_dynamic_call_one.begin
   :end-before: magictoken.ex_vectorize_dynamic_call_one.end
   :dedent: 12
   :linenos:

If this was a normal NumPy ufunc, we would have seen an exception
complaining that the ufunc couldn't handle the input types.  When we
call :func:`f` with integer arguments, not only do we receive an
answer, but we can verify that Numba created a loop supporting C
:code:`long` integers.

We can add additional loops by calling :func:`f` with different inputs:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_dynamic_call_two.begin
   :end-before: magictoken.ex_vectorize_dynamic_call_two.end
   :dedent: 12
   :linenos:

We can now verify that Numba added a second loop for dealing with
floating-point inputs, :code:`"dd->d"`.

If we mix input types to :func:`f`, we can verify that `NumPy ufunc
casting rules`_ are still in effect:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_dynamic_call_three.begin
   :end-before: magictoken.ex_vectorize_dynamic_call_three.end
   :dedent: 12
   :linenos:

.. _`NumPy ufunc casting rules`: http://docs.scipy.org/doc/numpy/reference/ufuncs.html#casting-rules

This example demonstrates that calling :func:`f` with mixed types
caused NumPy to select the floating-point loop, and cast the integer
argument to a floating-point value.  Thus, Numba did not create a
special :code:`"dl->d"` kernel.

This :class:`~numba.DUFunc` behavior leads us to a point similar to
the warning given above in "`The @vectorize decorator`_" subsection,
but instead of signature declaration order in the decorator, call
order matters.  If we had passed in floating-point arguments first,
any calls with integer arguments would be cast to double-precision
floating-point values.  For example:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_vectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_vectorize_dynamic_call_four.begin
   :end-before: magictoken.ex_vectorize_dynamic_call_four.end
   :dedent: 12
   :linenos:

If you require precise support for various type signatures, you should
specify them in the :func:`~numba.vectorize` decorator, and not rely
on dynamic compilation.

Dynamic generalized universal functions
=======================================

Similar to a dynamic universal function, if you do not specify any types to
the :func:`~numba.guvectorize` decorator, your Python function will be used
to build a dynamic generalized universal function, or :class:`~numba.GUFunc`.
For example:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_dynamic.begin
   :end-before: magictoken.ex_guvectorize_dynamic.end
   :dedent: 12
   :linenos:

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
compilation works for a :class:`~numba.GUFunc`:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_dynamic_call_one.begin
   :end-before: magictoken.ex_guvectorize_dynamic_call_one.end
   :dedent: 12
   :linenos:

If this was a normal :func:`guvectorize` function, we would have seen an
exception complaining that the ufunc could not handle the given input types.
When we call :func:`g()` with the input arguments, numba creates a new loop
for the input types.

We can add additional loops by calling :func:`g` with new arguments:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_dynamic_call_two.begin
   :end-before: magictoken.ex_guvectorize_dynamic_call_two.end
   :dedent: 12
   :linenos:

We can now verify that Numba added a second loop for dealing with
floating-point inputs, :code:`"dd->d"`.

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_dynamic_call_three.begin
   :end-before: magictoken.ex_guvectorize_dynamic_call_three.end
   :dedent: 12
   :linenos:

One can also verify that NumPy ufunc casting rules are working as expected:

.. literalinclude:: ../../../numba/tests/doc_examples/test_examples.py
   :language: python
   :caption: from ``test_guvectorize_dynamic`` of ``numba/tests/doc_examples/test_examples.py``
   :start-after: magictoken.ex_guvectorize_dynamic_call_four.begin
   :end-before: magictoken.ex_guvectorize_dynamic_call_four.end
   :dedent: 12
   :linenos:

If you need precise support for various type signatures, you should not rely on dynamic
compilation and instead, specify the types them as first
argument in the :func:`~numba.guvectorize` decorator.
