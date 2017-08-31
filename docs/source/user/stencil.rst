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
with :ref:`the parallel jit option <parallel_jit_option>` enables higher performance through
parallelism of the stencil execution.


Basic usage
===========

An example usage of the ``@stencil`` decorator::
 
   from numba import stencil

   @stencil
   def kernel1(a):
       return 0.25 * (a[0,1] + a[1,0] + a[0,-1] + a[-1,0])

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
of the output array being processed.  For example, if element ``(x,y)``
is being processed then ``a[0,0]`` in the stencil kernel corresponds to 
``a[x+0,y+0]`` in the input array.  Similarly, ``a[-1,1]`` in the stencil
kernel corresponds to ``a[x-1,y+1]`` in the input array.

Depending on the specified kernel, the kernel cannot be applied to the
borders of the output image as this may cause the input array to be
accessed out-of-bounds.  The stencil decorator detects when it is
unsafe to execute the kernel along the borders of the array and sets
those elements of the output array to zero.

To invoke a stencil on an input array, call the stencil as if it were
a regular function and pass the input array as the argument::

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
of the specified stencil kernel is ``float`` and has thus created the
output array as ``float`` while the input array is of type ``int``.

.. _stencil-kernel-shape-inference:

Kernel shape inference and border handling
==========================================

In the above example and in most cases, the array indexing in the 
stencil kernel will exclusively use Integer literals.
In such cases, the stencil decorator is able to analyze the stencil
kernel to determine its size.  In the above example, the stencil
decorator determines that the kernel is 3x3 since indices -1 to 1
are used for both the first and second dimensions.  Note that
the stencil decorator also correctly handles non-symmetric and 
non-square stencil kernels.

Based on the size of the stencil kernel, the stencil decorator is
able to compute the size of the border in the output array.  If
applying the kernel to some element of input array would cause
an index to be out-of-bounds then that element belongs to the border
of the output array.  In the above example, points -1 and +1 are
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
exclusively with Integer literals.  For example, let us say we
would like to compute the trailing 30-day moving average of a
time series of data.  One could write 
``(a[-29] + a[-28] + ... + a[-1] + a[0]) / 30`` but the stencil
decorator offers a more concise form using the ``neighborhood``
option::

   @stencil(neighborhood=((-29,0),))
   def kernel2(a):
       cumul = 0
       for i in range(-29,1):
           cumul += a[i]
       return cumul / 30

The neighborhood option is a tuple of tuples.  The outer tuple's
length is equal to the number of dimensions of the input array.
The inner tuples' lengths are always 2 because
each element of the outer tuple corresponds to minimum and
maximum index offsets used in the corresponding dimension.

If a user specifies a neighborhood but then in the kernel 
accesses elements outside the specified neighborhood, the behavior
is undefined.

Stencil invocation options
==========================

Internally, the stencil decorator transforms the specified stencil
kernel into a regular Python function.  In this process, the decorator
adds one optional argument in addition to the first required parameter, 
which is the input array.

.. _stencil-function-out:

``out``
-------

The optional ``out`` parameter is added to every stencil function
generated by Numba.  If specified, the ``out`` parameter tells 
Numba that the user is providing their own pre-allocated array 
to be used for the output of the stencil.  In this case, the
stencil function will not allocate its own output array.
An example usage is shown below::

   >>> import numpy as np
   >>> input_arr = np.arange(100).reshape((10, 10))
   >>> output_arr = np.full(input_arr.shape, 0.0)
   >>> kernel1(input_arr, out=output_arr)
