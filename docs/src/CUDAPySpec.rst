CUDA Python Specification (v0.1)
================================

(This documents reflects the implementation of CUDA-Python in NumbaPro 0.12.  In time, we may refine the specification.)

As CUDA python is becoming more mature, it has become necessary to define a formal specification for CUDA Python and the mapping to the PTX ISA.  There are places where the semantic of CUDA Python differs from the Python semantic.  The change in semantic is necessary for us to generate high-performance code that is otherwise hard to achieve.

No-Python Mode (NPM)
---------------------

CUDA Python is a superset of the No-Python mode (NPM).  NPM is a statically typed subset of the Python language.  It only supports lower level types; such as booleans, ints, floats, complex numbers and arrays.  It does not support Python objects.  Since we drop the support for objects entirely, many basic language construct must be handled differently.  For instance, a simple for-loop is::

    for i in range(10):
        ...
        
where range returns an iteratable.  NPM restricts the language so that only ``range`` or ``xrange`` can be used.  

For array support, NPM models the NumPy ndarray.  An array is a structure with a pointer to the data, an array of shape and an array of strides.  Valid attributes are ``shape``, ``strides``, ``size`` and ``ndim``.  Arrays cannot be unpacked.  The only way to access array elements is through the ``__getitem__`` and ``__setitem__`` operators (e.g. ``ary[i, j]``).  Slicing is not supported.  When indexing into an array, a N-dimension array must be provided with N indices.

Tuples are minimally supported for unpacking array ``shape`` and ``strides`` attributes and some return value of calls.

In time, we aim to enhance NPM to expand the supported subset and recognize more idiomatic Python patterns.

**Summary**:

* no object;
* no exception;
* for-loop only works on ``range`` or ``xrange``;
* supported types: ints, floats, complex numbers, and arrays.

Type Inference
----------------

The type inference algorithm for CUDA Python differs from Numba as we recognize that CUDA Python users require stronger typing to better predict code performance.  This is a summary of the type inference rules:

* Implicit coercion for all ints and floats only.
* Variable type is assigned at definition but a variable can be redefined; thus its type can be modified at the next assignment.
* Inside a loop, the variable type remains unchanged even at redefinition.  The type assigned at the preloop block (the dominator of the basic-block) is assumed.  This greatly differs from Python semantic.

User can force the type of any value by using the type object defined in ``numbapro`` namespace::

    from numbapro import cuda, int16, float32

    @cuda.autojit
    def a_cuda_kernel(arg):
        must_be_int16 = int16(123)
        must_be_float32 = float32(321)

Basic Arithmetic Operations
----------------------------

For binary operators ``+ - *``, the operands are coerced to the most generic type of the two before the computation.  The result type is the coerced type.

For floordiv ``//``, the coercion rule on operands for basic binary operators applies.  But, the result type is always coerced to an integer of the same bitwidth and at least has 32-bits.

For truediv ``/``, the operands are promoted to a floating point representation with bitwidth equals to the maximum of the two operands before the computation and at least has 32-bits.  The result type is the coerced type.

For binary bitwise ``& | ^ >> <<``, the operands must be of integer types and they are coerced to the most generic type of the two before the computation.  The result type is the coerced type.

For complex numbers, only ``+ - *`` are defined.

Please refer to the `CUDA-C Programming Guide <http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions>`_ for the precision each operation.

Intp
-----

``intp`` is used to represent the integer which width equals the address width.

Array Operations
------------------

Array attributes are read-only:

* ``shape`` contains the number of elements for each dimension.  It can be indexed or unpacked like a tuple.  It is a tuple of intp.
* ``strides`` contains the number of bytes skip to move forward to the next element for a given dimension.  It can be indexed or unpacked like a tuple.  It is a tuple of intp.
* ``size`` contains the number of elements in the array but may not be correspond to the actual size of the data buffer since strides can be zero or negative.  It is a intp.
* ``ndim`` contains the number of dimension in the array.  It is a intp.

``__getitem__`` returns the element at the given index.  Slicing or fancy indexing are not supported.  The result type is always the same as the element type of the array.

``__setitem__`` stores a value into the array at an index.  The value is coerced if necessary.

Math
-----

CUDA-Python translates math functions defined in the math module of the Python standard library.  All the function uses semantic of the CUDA-C definition.  Please refer to the `CUDA-C Programming Guide <http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix>`_.

Supported functions::

    math.acos
    math.asin
    math.atan
    math.arctan
    math.acosh
    math.asinh
    math.atanh
    math.cos
    math.sin
    math.tan
    math.cosh
    math.sinh
    math.tanh
    math.atan2
    math.exp
    math.expm1              # not available in python 2.6
    math.fabs
    math.log
    math.log10
    math.log1p
    math.sqrt
    math.pow
    math.ceil
    math.floor
    math.copysign
    math.fmod
    math.isnan
    math.isinf
    