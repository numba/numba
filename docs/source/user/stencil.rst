.. Copyright (c) 2017 Intel Corporation
   SPDX-License-Identifier: BSD-2-Clause

.. _numba-stencil:

================================
Using the ``@stencil`` decorator
================================

Stencils are a common computational pattern in which array elements 
are updated according to some fixed pattern called the stencil kernel.
Numba provides the ``@stencil`` decorator so that users may
easily specify a stencil kernel and Numba then generates the looping
code necessary to apply that kernel to some input array.  Thus, the
stencil decorator allows clearer, more concise code and in conjunction
with :ref:`the parallel jit option <parallel_jit_option>` enables higher
performance through parallelization of the stencil execution.


Basic usage
===========

An example use of the ``@stencil`` decorator::
 
   from numba import stencil

   @stencil
   def kernel1(a):
       return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])

The stencil kernel is specified by what looks like a standard Python
function definition but there are different semantics with
respect to array indexing.
Stencils produce an output array of the same size and shape as the
input array although depending on the kernel definition may have a
different type.
Conceptually, the stencil kernel is run once for each element in the
output array.  The return value from the stencil kernel is the value
written into the output array for that particular element.

The parameter ``a`` represents the input array over which the 
kernel is applied.  
Indexing into this array takes place with respect to the current element
of the output array being processed.  For example, if element ``(x, y)``
is being processed then ``a[0, 0]`` in the stencil kernel corresponds to
``a[x + 0, y + 0]`` in the input array.  Similarly, ``a[-1, 1]`` in the stencil
kernel corresponds to ``a[x - 1, y + 1]`` in the input array.

Depending on the specified kernel, the kernel may not be applicable to the
borders of the output array as this may cause the input array to be
accessed out-of-bounds.  The way in which the stencil decorator handles 
this situation is dependent upon which :ref:`stencil-mode` is selected.  
The default mode is for the stencil decorator to set the border elements 
of the output array to zero.

To invoke a stencil on an input array, call the stencil as if it were
a regular function and pass the input array as the argument. For example, using
the kernel defined above::

   >>> import numpy as np
   >>> input_arr = np.arange(100).reshape((10, 10))
   array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
          [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
          [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
          [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
          [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
          [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
          [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
          [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
          [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
          [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]])
   >>> output_arr = kernel1(input_arr)
   array([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
          [  0.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,   0.],
          [  0.,  21.,  22.,  23.,  24.,  25.,  26.,  27.,  28.,   0.],
          [  0.,  31.,  32.,  33.,  34.,  35.,  36.,  37.,  38.,   0.],
          [  0.,  41.,  42.,  43.,  44.,  45.,  46.,  47.,  48.,   0.],
          [  0.,  51.,  52.,  53.,  54.,  55.,  56.,  57.,  58.,   0.],
          [  0.,  61.,  62.,  63.,  64.,  65.,  66.,  67.,  68.,   0.],
          [  0.,  71.,  72.,  73.,  74.,  75.,  76.,  77.,  78.,   0.],
          [  0.,  81.,  82.,  83.,  84.,  85.,  86.,  87.,  88.,   0.],
          [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]])
   >>> input_arr.dtype
   dtype('int64')
   >>> output_arr.dtype
   dtype('float64')

Note that the stencil decorator has determined that the output type
of the specified stencil kernel is ``float64`` and has thus created the
output array as ``float64`` while the input array is of type ``int64``.

Stencil Parameters
==================

Stencil kernel definitions may take any number of arguments with
the following provisions.  The first argument must be an array.
The size and shape of the output array will be the same as that of the
first argument.  Additional arguments may either be scalars or
arrays.  For array arguments, those arrays must be at least as large
as the first argument (array) in each dimension.  Array indexing is relative for
all such input array arguments.

.. _stencil-kernel-shape-inference:

Kernel shape inference and border handling
==========================================

In the above example and in most cases, the array indexing in the 
stencil kernel will exclusively use ``Integer`` literals.
In such cases, the stencil decorator is able to analyze the stencil
kernel to determine its size.  In the above example, the stencil
decorator determines that the kernel is ``3 x 3`` in shape since indices
``-1`` to ``1`` are used for both the first and second dimensions.  Note that
the stencil decorator also correctly handles non-symmetric and 
non-square stencil kernels.

Based on the size of the stencil kernel, the stencil decorator is
able to compute the size of the border in the output array.  If
applying the kernel to some element of input array would cause
an index to be out-of-bounds then that element belongs to the border
of the output array.  In the above example, points ``-1`` and ``+1`` are
accessed in each dimension and thus the output array has a border
of size one in all dimensions.


Stencil decorator options
=========================

While the stencil decorator may be augmented in the future to 
provide additional mechanisms for border handling, at the moment
the stencil decorator currently supports only one option.

.. _stencil-neighborhood:

``neighborhood``
----------------

Sometimes it may be inconvenient to write the stencil kernel
exclusively with ``Integer`` literals.  For example, let us say we
would like to compute the trailing 30-day moving average of a
time series of data.  One could write 
``(a[-29] + a[-28] + ... + a[-1] + a[0]) / 30`` but the stencil
decorator offers a more concise form using the ``neighborhood``
option::

   @stencil(neighborhood = ((-29, 0),))
   def kernel2(a):
       cumul = 0
       for i in range(-29, 1):
           cumul += a[i]
       return cumul / 30

The neighborhood option is a tuple of tuples.  The outer tuple's
length is equal to the number of dimensions of the input array.
The inner tuple's lengths are always two because
each element of the outer tuple corresponds to minimum and
maximum index offsets used in the corresponding dimension.

If a user specifies a neighborhood but the kernel accesses elements outside the
specified neighborhood, **the behavior is undefined.**

.. _stencil-mode:

``mode``
--------

The optional mode parameter controls how the border of the output array
is handled.  Currently, there is only one supported value, ``"constant"``.
In ``constant`` mode, the stencil kernel is not applied in cases where
the kernel would access elements outside the valid range of the input
array.  In such cases, those elements in the output array are assigned
to a constant value, as specified by the ``cval`` parameter.

``cval``
--------

The optional cval parameter defaults to zero but can be set to any
desired value, which is then used for the border of the output array
if the mode parameter is set to ``constant``.  The cval parameter is 
ignored in all other modes.  The type of the cval parameter must match
the return type of the stencil kernel.  If the user wishes the output
array to be constructed from a particular type then they should ensure
that the stencil kernel returns that type.

``standard_indexing``
---------------------

By default, all array accesses in a stencil kernel are processed as
relative indices as described above.  However, sometimes it may be
advantageous to pass an auxiliary array (e.g. an array of weights)
to a stencil kernel and have that array use standard Python indexing
rather than relative indexing.  For this purpose, there is the
stencil decorator option ``standard_indexing`` whose value is a
collection of strings whose names match those parameters to the
stencil function that are to be accessed with standard Python indexing
rather than relative indexing::

    @stencil(standard_indexing=("b",))
    def kernel3(a, b):
        return a[-1] * b[0] + a[0] + b[1]

``StencilFunc``
===============

The stencil decorator returns a callable object of type ``StencilFunc``.
``StencilFunc`` objects contains a number of attributes but the only one of
potential interest to users is the ``neighborhood`` attribute.
If the ``neighborhood`` option was passed to the stencil decorator then
the provided neighborhood is stored in this attribute.  Else, upon 
first execution or compilation, the system calculates the neighborhood
as described above and then stores the computed neighborhood into this
attribute.  A user may then inspect the attribute if they wish to verify
that the calculated neighborhood is correct.

Stencil invocation options
==========================

Internally, the stencil decorator transforms the specified stencil
kernel into a regular Python function.  This function will have the
same parameters as specified in the stencil kernel definition but will
also include the following optional parameter.

.. _stencil-function-out:

``out``
-------

The optional ``out`` parameter is added to every stencil function
generated by Numba.  If specified, the ``out`` parameter tells 
Numba that the user is providing their own pre-allocated array 
to be used for the output of the stencil.  In this case, the
stencil function will not allocate its own output array.
Users should assure that the return type of the stencil kernel can
be safely cast to the element-type of the user-specified output array
following the `Numpy ufunc casting rules`_.

.. _`Numpy ufunc casting rules`: http://docs.scipy.org/doc/numpy/reference/ufuncs.html#casting-rules

An example usage is shown below::

   >>> import numpy as np
   >>> input_arr = np.arange(100).reshape((10, 10))
   >>> output_arr = np.full(input_arr.shape, 0.0)
   >>> kernel1(input_arr, out=output_arr)
