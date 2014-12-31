
========================
Supported Numpy features
========================

One objective of Numba is having a seamless integration with `NumPy`_.
NumPy arrays provide an efficient storage method for homogeneous sets of
data.  NumPy dtypes provide type information useful when compiling, and
the regular, structured storage of potentially large amounts of data
in memory provides an ideal memory layout for code generation.  Numba
excels at generating code that executes on top of NumPy arrays.

NumPy support in Numba comes in many forms:

* Numba understands calls to NumPy `ufuncs`_ and is able to generate
  equivalent native code for many of them.

* NumPy arrays are directly supported in Numba.  Access to Numpy arrays
  is very efficient, as indexing is lowered to direct memory accesses
  when possible.

* Numba is able to generate `ufuncs`_ and `gufuncs`_. This means that it
  is possible to implement ufuncs and gufuncs within Python, getting
  speeds comparable to that of ufuncs/gufuncs implemented in C extension
  modules using the NumPy C API.

.. _NumPy: http://www.numpy.org/
.. _ufuncs: http://docs.scipy.org/doc/numpy/reference/ufuncs.html
.. _gufuncs: http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html


Scalar types
============

Numba supports the following Numpy scalar types:

* **Integers**: all integers of either signedness, and any width up to 64 bits
* **Booleans**
* **Real numbers:** single-precision (32-bit) and double-precision (64-bit) reals
* **Complex numbers:** single-precision (2x32-bit) and double-precision (2x64-bit) complex numbers
* **Datetimes and timestamps:** of any unit
* **Structured scalars:** structured scalars made of any of the types above
* **Character sequences** (but no operations are available on them)

The following scalar types and features are not supported:

* **Arbitrary Python objects**
* **Half-precision and extended-precision** real and complex numbers

.. seealso::
   `Numpy scalars <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`_
   reference.


Array types
===========

Arrays of any of the scalar types above are supported, regardless of the shape
or layout.


Standard ufuncs
===============

One objective of Numba is having all the
`standard ufuncs in NumPy <http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs>`_
understood by Numba.  When a supported ufunc is found when compiling a
function, Numba maps the ufunc to equivalent native code.  This allows the
use of those ufuncs in Numba code that gets compiled in :term:`nopython mode`.

Limitations
-----------

Right now, only a selection of the standard ufuncs work in :term:`nopython mode`.

Also, in its current implementation ufuncs working on arrays will only
compile in nopython mode if their output array is passed explicitly.
This limitation does not apply when working with scalars.

Following is a list of the different standard ufuncs that Numba is aware of,
sorted in the same way as in the NumPy documentation.


Math operations
---------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 add                 Yes          Yes
 subtract            Yes          Yes
 multiply            Yes          Yes
 divide              Yes          Yes
 logaddexp           Yes          Yes
 logaddexp2          Yes          Yes
 true_divide         Yes          Yes
 floor_divide        Yes          Yes
 negative            Yes          Yes
 power               Yes          Yes
 remainder           Yes          Yes
 mod                 Yes          Yes
 fmod                Yes          Yes
 abs                 Yes          Yes
 absolute            Yes          Yes
 fabs                Yes          Yes
 rint                Yes          Yes
 sign                Yes          Yes
 conj                Yes          Yes
 exp                 Yes          Yes
 exp2                Yes          Yes
 log                 Yes          Yes
 log2                Yes          Yes
 log10               Yes          Yes
 expm1               Yes          Yes
 log1p               Yes          Yes
 sqrt                Yes          Yes
 square              Yes          Yes
 reciprocal          Yes          Yes
 conjugate           Yes          Yes
==============  =============  ===========


Trigonometric functions
-----------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 sin                 Yes          Yes
 cos                 Yes          Yes
 tan                 Yes          Yes
 arcsin              Yes          Yes
 arccos              Yes          Yes
 arctan              Yes          Yes
 arctan2             Yes          Yes
 hypot               Yes          Yes
 sinh                Yes          Yes
 cosh                Yes          Yes
 tanh                Yes          Yes
 arcsinh             Yes          Yes
 arccosh             Yes          Yes
 arctanh             Yes          Yes
 deg2rad             Yes          Yes
 rad2deg             Yes          Yes
 degrees             Yes          Yes
 radians             Yes          Yes
==============  =============  ===========


Bit-twiddling functions
-----------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 bitwise_and         Yes          Yes
 bitwise_or          Yes          Yes
 bitwise_xor         Yes          Yes
 bitwise_not         Yes          Yes
 invert              Yes          Yes
 left_shift          Yes          Yes
 right_shift         Yes          Yes
==============  =============  ===========


Comparison functions
--------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 greater             Yes          Yes
 greater_equal       Yes          Yes
 less                Yes          Yes
 less_equal          Yes          Yes
 not_equal           Yes          Yes
 equal               Yes          Yes
 logical_and         Yes          Yes
 logical_or          Yes          Yes
 logical_xor         Yes          Yes
 logical_not         Yes          Yes
 maximum             Yes          Yes
 minimum             Yes          Yes
 fmax                Yes          Yes
 fmin                Yes          Yes
==============  =============  ===========


Floating functions
------------------

==============  =============  ===========
    UFUNC                  MODE
--------------  --------------------------
    name         python-mode    no-python
==============  =============  ===========
 isfinite            Yes          Yes
 isinf               Yes          Yes
 isnan               Yes          Yes
 signbit             Yes          Yes
 copysign            Yes          Yes
 nextafter           Yes          Yes
 modf                Yes          No
 ldexp               Yes (*)      Yes
 frexp               Yes          No
 floor               Yes          Yes
 ceil                Yes          Yes
 trunc               Yes          Yes
 spacing             Yes          Yes
==============  =============  ===========

(\*) not supported on windows 32 bit
