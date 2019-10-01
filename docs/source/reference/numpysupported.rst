
.. _numpy-support:

========================
Supported NumPy features
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

The following sections focus on the Numpy features supported in
:term:`nopython mode`, unless otherwise stated.


Scalar types
============

Numba supports the following Numpy scalar types:

* **Integers**: all integers of either signedness, and any width up to 64 bits
* **Booleans**
* **Real numbers:** single-precision (32-bit) and double-precision (64-bit) reals
* **Complex numbers:** single-precision (2x32-bit) and double-precision (2x64-bit) complex numbers
* **Datetimes and timestamps:** of any unit
* **Character sequences** (but no operations are available on them)
* **Structured scalars:** structured scalars made of any of the types above and arrays of the types above

The following scalar types and features are not supported:

* **Arbitrary Python objects**
* **Half-precision and extended-precision** real and complex numbers
* **Nested structured scalars** the fields of structured scalars may not contain other structured scalars

The operations supported on NumPy scalars are almost the same as on the
equivalent built-in types such as ``int`` or ``float``.  You can use a type's
constructor to convert from a different type or width. In addition you can use
the ``view(np.<dtype>)`` method to bitcast all ``int`` and ``float`` types
within the same width. However, you must define the scalar using a NumPy
constructor within a jitted function. For example, the following will work:

.. code:: pycon

    >>> import numpy as np
    >>> from numba import njit
    >>> @njit
    ... def bitcast():
    ...     i = np.int64(-1)
    ...     print(i.view(np.uint64))
    ...
    >>> bitcast()
    18446744073709551615


Whereas the following will not work:


.. code:: pycon

    >>> import numpy as np
    >>> from numba import njit
    >>> @njit
    ... def bitcast(i):
    ...     print(i.view(np.uint64))
    ...
    >>> bitcast(np.int64(-1))
    ---------------------------------------------------------------------------
    TypingError                               Traceback (most recent call last)
        ...
    TypingError: Failed in nopython mode pipeline (step: ensure IR is legal prior to lowering)
    'view' can only be called on NumPy dtypes, try wrapping the variable with 'np.<dtype>()'

    File "<ipython-input-3-fc40aaab84c4>", line 3:
    def bitcast(i):
        print(i.view(np.uint64))

Structured scalars support attribute getting and setting, as well as
member lookup using constant strings.

.. seealso::
   `Numpy scalars <http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html>`_
   reference.


Array types
===========

`Numpy arrays <http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`_
of any of the scalar types above are supported, regardless of the shape
or layout.

Array access
------------

Arrays support normal iteration.  Full basic indexing and slicing is
supported.  A subset of advanced indexing is also supported: only one
advanced index is allowed, and it has to be a one-dimensional array
(it can be combined with an arbitrary number of basic indices as well).

.. seealso::
   `Numpy indexing <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_
   reference.

Attributes
----------

The following attributes of Numpy arrays are supported:

* :attr:`~numpy.ndarray.dtype`
* :attr:`~numpy.ndarray.flags`
* :attr:`~numpy.ndarray.flat`
* :attr:`~numpy.ndarray.itemsize`
* :attr:`~numpy.ndarray.ndim`
* :attr:`~numpy.ndarray.shape`
* :attr:`~numpy.ndarray.size`
* :attr:`~numpy.ndarray.strides`
* :attr:`~numpy.ndarray.T`
* :attr:`~numpy.ndarray.real`
* :attr:`~numpy.ndarray.imag`

The ``flags`` object
''''''''''''''''''''

The object returned by the :attr:`~numpy.ndarray.flags` attribute supports
the ``contiguous``, ``c_contiguous`` and ``f_contiguous`` attributes.

The ``flat`` object
'''''''''''''''''''

The object returned by the :attr:`~numpy.ndarray.flat` attribute supports
iteration and indexing, but be careful: indexing is very slow on
non-C-contiguous arrays.

The ``real`` and ``imag`` attributes
''''''''''''''''''''''''''''''''''''

Numpy supports these attributes regardless of the dtype but Numba chooses to
limit their support to avoid potential user error.  For numeric dtypes,
Numba follows Numpy's behavior.  The :attr:`~numpy.ndarray.real` attribute
returns a view of the real part of the complex array and it behaves as an identity
function for other numeric dtypes.  The :attr:`~numpy.ndarray.imag` attribute
returns a view of the imaginary part of the complex array and it returns a zero
array with the same shape and dtype for other numeric dtypes.  For non-numeric
dtypes, including all structured/record dtypes, using these attributes will
result in a compile-time (`TypingError`) error.  This behavior differs from
Numpy's but it is chosen to avoid the potential confusion with field names that
overlap these attributes.

Calculation
-----------

The following methods of Numpy arrays are supported in their basic form
(without any optional arguments):

* :meth:`~numpy.ndarray.all`
* :meth:`~numpy.ndarray.any`
* :meth:`~numpy.ndarray.argmax`
* :meth:`~numpy.ndarray.argmin`
* :meth:`~numpy.ndarray.conj`
* :meth:`~numpy.ndarray.conjugate`
* :meth:`~numpy.ndarray.cumprod`
* :meth:`~numpy.ndarray.cumsum`
* :meth:`~numpy.ndarray.max`
* :meth:`~numpy.ndarray.mean`
* :meth:`~numpy.ndarray.min`
* :meth:`~numpy.ndarray.nonzero`
* :meth:`~numpy.ndarray.prod`
* :meth:`~numpy.ndarray.std`
* :meth:`~numpy.ndarray.take`
* :meth:`~numpy.ndarray.var`

The corresponding top-level Numpy functions (such as :func:`numpy.prod`)
are similarly supported.

Other methods
-------------

The following methods of Numpy arrays are supported:

* :meth:`~numpy.ndarray.argsort` (``kind`` key word argument supported for
  values ``'quicksort'`` and ``'mergesort'``)
* :meth:`~numpy.ndarray.astype` (only the 1-argument form)
* :meth:`~numpy.ndarray.copy` (without arguments)
* :meth:`~numpy.ndarray.dot` (only the 1-argument form)
* :meth:`~numpy.ndarray.flatten` (no order argument; 'C' order only)
* :meth:`~numpy.ndarray.item` (without arguments)
* :meth:`~numpy.ndarray.itemset` (only the 1-argument form)
* :meth:`~numpy.ndarray.ravel` (no order argument; 'C' order only)
* :meth:`~numpy.ndarray.repeat` (no axis argument)
* :meth:`~numpy.ndarray.reshape` (only the 1-argument form)
* :meth:`~numpy.ndarray.sort` (without arguments)
* :meth:`~numpy.ndarray.sum` (with or without the ``axis`` and/or ``dtype``
  arguments.)

  * ``axis`` only supports ``integer`` values.
  * If the ``axis`` argument is a compile-time constant, all valid values
    are supported.
    An out-of-range value will result in a ``LoweringError`` at compile-time.
  * If the ``axis`` argument is not a compile-time constant, only values
    from 0 to 3 are supported.
    An out-of-range value will result in a runtime exception.
  * All numeric ``dtypes`` are supported in the ``dtype`` parameter.
    ``timedelta`` arrays can be used as input arrays but ``timedelta`` is not
    supported as ``dtype`` parameter.
  * When a ``dtype`` is given, it determines the type of the internal
    accumulator. When it is not, the selection is made automatically based on
    the input array's ``dtype``, mostly following the same rules as NumPy.
    However, on 64-bit Windows, Numba uses a 64-bit accumulator for integer
    inputs (``int64`` for ``int32`` inputs and ``uint64`` for ``uint32``
    inputs), while NumPy would use a 32-bit accumulator in those cases.


* :meth:`~numpy.ndarray.transpose`
* :meth:`~numpy.ndarray.view` (only the 1-argument form)


.. warning::
   Sorting may be slightly slower than Numpy's implementation.


Functions
=========

Linear algebra
--------------

Basic linear algebra is supported on 1-D and 2-D contiguous arrays of
floating-point and complex numbers:

* :func:`numpy.dot`
* :func:`numpy.kron` ('C' and 'F' order only)
* :func:`numpy.outer`
* :func:`numpy.trace` (only the first argument).
* :func:`numpy.vdot`
* On Python 3.5 and above, the matrix multiplication operator from
  :pep:`465` (i.e. ``a @ b`` where ``a`` and ``b`` are 1-D or 2-D arrays).
* :func:`numpy.linalg.cholesky`
* :func:`numpy.linalg.cond` (only non string values in ``p``).
* :func:`numpy.linalg.det`
* :func:`numpy.linalg.eig` (only running with data that does not cause a domain
  change is supported e.g. real input -> real
  output, complex input -> complex output).
* :func:`numpy.linalg.eigh` (only the first argument).
* :func:`numpy.linalg.eigvals` (only running with data that does not cause a
  domain change is supported e.g. real input -> real output,
  complex input -> complex output).
* :func:`numpy.linalg.eigvalsh` (only the first argument).
* :func:`numpy.linalg.inv`
* :func:`numpy.linalg.lstsq`
* :func:`numpy.linalg.matrix_power`
* :func:`numpy.linalg.matrix_rank`
* :func:`numpy.linalg.norm` (only the 2 first arguments and only non string
  values in ``ord``).
* :func:`numpy.linalg.pinv`
* :func:`numpy.linalg.qr` (only the first argument).
* :func:`numpy.linalg.slogdet`
* :func:`numpy.linalg.solve`
* :func:`numpy.linalg.svd` (only the 2 first arguments).

.. note::
   The implementation of these functions needs Scipy 0.16+ to be installed.

Reductions
----------

The following reduction functions are supported:

* :func:`numpy.diff` (only the 2 first arguments)
* :func:`numpy.median` (only the first argument)
* :func:`numpy.nancumprod` (only the first argument, requires NumPy >= 1.12))
* :func:`numpy.nancumsum` (only the first argument, requires NumPy >= 1.12))
* :func:`numpy.nanmax` (only the first argument)
* :func:`numpy.nanmean` (only the first argument)
* :func:`numpy.nanmedian` (only the first argument)
* :func:`numpy.nanmin` (only the first argument)
* :func:`numpy.nanpercentile` (only the 2 first arguments,
  requires NumPy >= 1.11, complex dtypes unsupported)
* :func:`numpy.nanquantile` (only the 2 first arguments, requires NumPy >= 1.15,
  complex dtypes unsupported)
* :func:`numpy.nanprod` (only the first argument)
* :func:`numpy.nanstd` (only the first argument)
* :func:`numpy.nansum` (only the first argument)
* :func:`numpy.nanvar` (only the first argument)
* :func:`numpy.percentile` (only the 2 first arguments, requires NumPy >= 1.10,
  complex dtypes unsupported)
* :func:`numpy.quantile` (only the 2 first arguments, requires NumPy >= 1.15,
  complex dtypes unsupported)

Other functions
---------------

The following top-level functions are supported:

* :func:`numpy.append`
* :func:`numpy.arange`
* :func:`numpy.argsort` (``kind`` key word argument supported for values
  ``'quicksort'`` and ``'mergesort'``)
* :func:`numpy.array` (only the 2 first arguments)
* :func:`numpy.array_equal`
* :func:`numpy.asarray` (only the 2 first arguments)
* :func:`numpy.asfortranarray` (only the first argument)
* :func:`numpy.atleast_1d`
* :func:`numpy.atleast_2d`
* :func:`numpy.atleast_3d`
* :func:`numpy.bartlett`
* :func:`numpy.bincount` (only the 2 first arguments)
* :func:`numpy.blackman`
* :func:`numpy.column_stack`
* :func:`numpy.concatenate`
* :func:`numpy.convolve` (only the 2 first arguments)
* :func:`numpy.copy` (only the first argument)
* :func:`numpy.corrcoef` (only the 3 first arguments, requires NumPy >= 1.10 and
  SciPy >= 0.16; extreme value handling per NumPy 1.11+)
* :func:`numpy.correlate` (only the 2 first arguments)
* :func:`numpy.count_nonzero` (axis only supports scalar values)
* :func:`numpy.cov` (only the 5 first arguments, requires NumPy >= 1.10 and SciPy >= 0.16)
* :func:`numpy.cross` (only the 2 first arguments; at least one of the input
  arrays should have ``shape[-1] == 3``)

  * If ``shape[-1] == 2`` for both inputs, please replace your
    :func:`numpy.cross` call with :func:`numba.numpy_extensions.cross2d`.

* :func:`numpy.delete` (only the 2 first arguments)
* :func:`numpy.diag`
* :func:`numpy.digitize`
* :func:`numpy.dstack`
* :func:`numpy.dtype` (only the first argument)
* :func:`numpy.ediff1d`
* :func:`numpy.empty` (only the 2 first arguments)
* :func:`numpy.empty_like` (only the 2 first arguments)
* :func:`numpy.expand_dims`
* :func:`numpy.extract`
* :func:`numpy.eye`
* :func:`numpy.fill_diagonal`
* :func:`numpy.flatten` (no order argument; 'C' order only)
* :func:`numpy.flatnonzero`
* :func:`numpy.frombuffer` (only the 2 first arguments)
* :func:`numpy.full` (only the 3 first arguments)
* :func:`numpy.full_like` (only the 3 first arguments)
* :func:`numpy.hamming`
* :func:`numpy.hanning`
* :func:`numpy.histogram` (only the 3 first arguments)
* :func:`numpy.hstack`
* :func:`numpy.identity`
* :func:`numpy.kaiser`
* :func:`numpy.interp` (only the 3 first arguments; requires NumPy >= 1.10)
* :func:`numpy.linspace` (only the 3-argument form)
* :class:`numpy.ndenumerate`
* :class:`numpy.ndindex`
* :class:`numpy.nditer` (only the first argument)
* :func:`numpy.ones` (only the 2 first arguments)
* :func:`numpy.ones_like` (only the 2 first arguments)
* :func:`numpy.partition` (only the 2 first arguments)
* :func:`numpy.ptp` (only the first argument)
* :func:`numpy.ravel` (no order argument; 'C' order only)
* :func:`numpy.repeat` (no axis argument)
* :func:`numpy.reshape` (no order argument; 'C' order only)
* :func:`numpy.roll` (only the 2 first arguments; second argument ``shift``
  must be an integer)
* :func:`numpy.roots`
* :func:`numpy.round_`
* :func:`numpy.searchsorted` (only the 3 first arguments)
* :func:`numpy.select` (only using homogeneous lists or tuples for the first
  two arguments, condlist and choicelist). Additionally, these two arguments
  can only contain arrays (unlike Numpy that also accepts tuples).
* :func:`numpy.shape`
* :func:`numpy.sinc`
* :func:`numpy.sort` (no optional arguments)
* :func:`numpy.stack`
* :func:`numpy.take` (only the 2 first arguments)
* :func:`numpy.transpose`
* :func:`numpy.trapz` (only the 3 first arguments)
* :func:`numpy.tri` (only the 3 first arguments; third argument ``k`` must be an integer)
* :func:`numpy.tril` (second argument ``k`` must be an integer)
* :func:`numpy.tril_indices` (all arguments must be integer)
* :func:`numpy.tril_indices_from` (second argument ``k`` must be an integer)
* :func:`numpy.triu` (second argument ``k`` must be an integer)
* :func:`numpy.triu_indices` (all arguments must be integer)
* :func:`numpy.triu_indices_from` (second argument ``k`` must be an integer)
* :func:`numpy.unique` (only the first argument)
* :func:`numpy.vander`
* :func:`numpy.vstack`
* :func:`numpy.where`
* :func:`numpy.zeros` (only the 2 first arguments)
* :func:`numpy.zeros_like` (only the 2 first arguments)

The following constructors are supported, both with a numeric input (to
construct a scalar) or a sequence (to construct an array):

* :class:`numpy.bool_`
* :class:`numpy.complex64`
* :class:`numpy.complex128`
* :class:`numpy.float32`
* :class:`numpy.float64`
* :class:`numpy.int8`
* :class:`numpy.int16`
* :class:`numpy.int32`
* :class:`numpy.int64`
* :class:`numpy.intc`
* :class:`numpy.intp`
* :class:`numpy.uint8`
* :class:`numpy.uint16`
* :class:`numpy.uint32`
* :class:`numpy.uint64`
* :class:`numpy.uintc`
* :class:`numpy.uintp`

The following machine parameter classes are supported, with all purely numerical
attributes:

* :class:`numpy.iinfo`
* :class:`numpy.finfo` (``machar`` attribute not supported)
* :class:`numpy.MachAr` (with no arguments to the constructor)


Literal arrays
--------------

.. XXX should this part of the user's guide?

Neither Python nor Numba has actual array literals, but you can construct
arbitrary arrays by calling :func:`numpy.array` on a nested tuple::

   a = numpy.array(((a, b, c), (d, e, f)))

(nested lists are not yet supported by Numba)


Modules
=======

.. _numpy-random:

``random``
----------

Numba supports top-level functions from the
`numpy.random <http://docs.scipy.org/doc/numpy/reference/routines.random.html>`_
module, but does not allow you to create individual RandomState instances.
The same algorithms are used as for :ref:`the standard
random module <pysupported-random>` (and therefore the same notes apply),
but with an independent internal state: seeding or drawing numbers from
one generator won't affect the other.

The following functions are supported.

Initialization
''''''''''''''

* :func:`numpy.random.seed`: with an integer argument only

Simple random data
''''''''''''''''''

* :func:`numpy.random.rand`
* :func:`numpy.random.randint` (only the first two arguments)
* :func:`numpy.random.randn`
* :func:`numpy.random.random`
* :func:`numpy.random.random_sample`
* :func:`numpy.random.ranf`
* :func:`numpy.random.sample`

Permutations
''''''''''''

* :func:`numpy.random.choice`: the optional *p* argument (probabilities
  array) is not supported
* :func:`numpy.random.permutation`
* :func:`numpy.random.shuffle`: the sequence argument must be a one-dimension
  Numpy array or buffer-providing object (such as a :class:`bytearray`
  or :class:`array.array`)

Distributions
'''''''''''''

.. warning:: The `size` argument is not supported in the following functions.

* :func:`numpy.random.beta`
* :func:`numpy.random.binomial`
* :func:`numpy.random.chisquare`
* :func:`numpy.random.exponential`
* :func:`numpy.random.f`
* :func:`numpy.random.gamma`
* :func:`numpy.random.geometric`
* :func:`numpy.random.gumbel`
* :func:`numpy.random.hypergeometric`
* :func:`numpy.random.laplace`
* :func:`numpy.random.logistic`
* :func:`numpy.random.lognormal`
* :func:`numpy.random.logseries`
* :func:`numpy.random.multinomial`
* :func:`numpy.random.negative_binomial`
* :func:`numpy.random.normal`
* :func:`numpy.random.pareto`
* :func:`numpy.random.poisson`
* :func:`numpy.random.power`
* :func:`numpy.random.rayleigh`
* :func:`numpy.random.standard_cauchy`
* :func:`numpy.random.standard_exponential`
* :func:`numpy.random.standard_gamma`
* :func:`numpy.random.standard_normal`
* :func:`numpy.random.standard_t`
* :func:`numpy.random.triangular`
* :func:`numpy.random.uniform`
* :func:`numpy.random.vonmises`
* :func:`numpy.random.wald`
* :func:`numpy.random.weibull`
* :func:`numpy.random.zipf`

.. note::
   Calling :func:`numpy.random.seed` from non-Numba code (or from
   :term:`object mode` code) will seed the Numpy random generator, not the
   Numba random generator.

.. note::
   Since version 0.28.0, the generator is thread-safe and fork-safe.  Each
   thread and each process will produce independent streams of random numbers.


``stride_tricks``
-----------------

The following function from the :mod:`numpy.lib.stride_tricks` module
is supported:

* :func:`~numpy.lib.stride_tricks.as_strided` (the *strides* argument
  is mandatory, the *subok* argument is not supported)

.. _supported_ufuncs:

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
Following is a list of the different standard ufuncs that Numba is aware of,
sorted in the same way as in the NumPy documentation.


Math operations
---------------

==============  =============  ===============
    UFUNC                  MODE
--------------  ------------------------------
    name         object mode    nopython mode
==============  =============  ===============
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
==============  =============  ===============


Trigonometric functions
-----------------------

==============  =============  ===============
    UFUNC                  MODE
--------------  ------------------------------
    name         object mode    nopython mode
==============  =============  ===============
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
==============  =============  ===============


Bit-twiddling functions
-----------------------

==============  =============  ===============
    UFUNC                  MODE
--------------  ------------------------------
    name         object mode    nopython mode
==============  =============  ===============
 bitwise_and         Yes          Yes
 bitwise_or          Yes          Yes
 bitwise_xor         Yes          Yes
 bitwise_not         Yes          Yes
 invert              Yes          Yes
 left_shift          Yes          Yes
 right_shift         Yes          Yes
==============  =============  ===============


Comparison functions
--------------------

==============  =============  ===============
    UFUNC                  MODE
--------------  ------------------------------
    name         object mode    nopython mode
==============  =============  ===============
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
==============  =============  ===============


Floating functions
------------------

==============  =============  ===============
    UFUNC                  MODE
--------------  ------------------------------
    name         object mode    nopython mode
==============  =============  ===============
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
==============  =============  ===============

(\*) not supported on windows 32 bit
