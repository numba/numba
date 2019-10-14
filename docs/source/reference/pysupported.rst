.. _pysupported:

=========================
Supported Python features
=========================

Apart from the :ref:`pysupported-language` part below, which applies to both
:term:`object mode` and :term:`nopython mode`, this page only lists the
features supported in :term:`nopython mode`.

.. warning::
    Numba behavior differs from Python semantics in some situations.  We
    strongly advise reviewing :ref:`pysemantics` to become familiar with these
    differences.


.. _pysupported-language:

Language
========

Constructs
----------

Numba strives to support as much of the Python language as possible, but
some language features are not available inside Numba-compiled functions. The following Python language features are not currently supported:

* Class definition
* Exception handling (``try .. except``, ``try .. finally``)
* Context management (the ``with`` statement)
* Some comprehensions (list comprehension is supported, but not dict, set or generator comprehensions)
* Generator delegation (``yield from``)

The ``raise`` statement is supported in several forms:

* ``raise`` (to re-raise the current exception)
* ``raise SomeException``
* ``raise SomeException(<arguments>)``: in :term:`nopython mode`, constructor
  arguments must be :term:`compile-time constants <compile-time constant>`

Similarly, the ``assert`` statement is supported with or without an error
message.

Functions
---------

Function calls
''''''''''''''

Numba supports function calls using positional and named arguments, as well
as arguments with default values and ``*args`` (note the argument for
``*args`` can only be a tuple, not a list).  Explicit ``**kwargs`` are
not supported.

Function calls to locally defined inner functions are supported as long as
they can be fully inlined.

Functions as arguments
''''''''''''''''''''''

Functions can be passed as argument into another function.  But, they cannot
be returned. For example:

.. code-block:: python

  from numba import jit

  @jit
  def add1(x):
      return x + 1

  @jit
  def bar(fn, x):
      return fn(x)

  @jit
  def foo(x):
      return bar(add1, x)

  # Passing add1 within numba compiled code.
  print(foo(1))
  # Passing add1 into bar from interpreted code
  print(bar(add1, 1))

.. note:: Numba does not handle function objects as real objects.  Once a
          function is assigned to a variable, the variable cannot be
          re-assigned to a different function.


Inner function and closure
'''''''''''''''''''''''''''

Numba now supports inner functions as long as they are non-recursive
and only called locally, but not passed as argument or returned as
result. The use of closure variables (variables defined in outer scopes)
within an inner function is also supported.

Recursive calls
'''''''''''''''

Most recursive call patterns are supported.  The only restriction is that the
recursive callee must have a control-flow path that returns without recursing.
Numba is able to type-infer recursive functions without specifying the function
type signature (which is required in numba 0.28 and earlier).
Recursive calls can even call into a different overload of the function.

.. XXX add reference to NBEP

Generators
----------

Numba supports generator functions and is able to compile them in
:term:`object mode` and :term:`nopython mode`.  The returned generator
can be used both from Numba-compiled code and from regular Python code.

Coroutine features of generators are not supported (i.e. the
:meth:`generator.send`, :meth:`generator.throw`, :meth:`generator.close`
methods).

.. _pysupported-builtin-types:

Built-in types
==============

int, bool
---------

Arithmetic operations as well as truth values are supported.

The following attributes and methods are supported:

* ``.conjugate()``
* ``.real``
* ``.imag``

float, complex
--------------

Arithmetic operations as well as truth values are supported.

The following attributes and methods are supported:

* ``.conjugate()``
* ``.real``
* ``.imag``

str
---

Numba supports (Unicode) strings in Python 3.  Strings can be passed into
:term:`nopython mode` as arguments, as well as constructed and returned from
:term:`nopython mode`. As in Python, slices (even of length 1) return a new,
reference counted string.  Optimized code paths for efficiently accessing
single characters may be introduced in the future.

The in-memory representation is the same as was introduced in Python 3.4, with
each string having a tag to indicate whether the string is using a 1, 2, or 4
byte character width in memory.  When strings of different encodings are
combined (as in concatenation), the resulting string automatically uses the
larger character width of the two input strings.  String slices also use the
same character width as the original string, even if the slice could be
represented with a narrower character width.  (These details are invisible to
the user, of course.)

The following functions, attributes and methods are currently supported:

* ``len()``
* ``+`` (concatenation of strings)
* ``*`` (repetition of strings)
* ``in``, ``.contains()``
* ``==``, ``<``, ``<=``, ``>``, ``>=`` (comparison)
* ``.startswith()``
* ``.endswith()``
* ``.find()``
* ``.center()``
* ``.ljust()``
* ``.rjust()``
* ``.split()``
* ``.join()``
* ``.lstrip()``
* ``.rstrip()``
* ``.strip()``
* ``.isupper()``
* ``.upper()``
* ``.zfill()``
* ``.count()``

Additional operations as well as support for Python 2 strings / Python 3 bytes
will be added in a future version of Numba.  Python 2 Unicode objects will
likely never be supported.

.. warning::
    The performance of some operations is known to be slower than the CPython
    implementation. These include substring search (``in``, ``.contains()``
    and ``find()``) and string creation (like ``.split()``).  Improving the
    string performance is an ongoing task, but the speed of CPython is
    unlikely to be surpassed for basic string operation in isolation.
    Numba is most successfuly used for larger algorithms that happen to
    involve strings, where basic string operations are not the bottleneck.


tuple
-----

The following operations are supported:

* tuple construction
* tuple unpacking
* comparison between tuples
* iteration and indexing over homogeneous tuples
* addition (concatenation) between tuples
* slicing tuples with a constant slice
* the index method on tuples

list
----


.. warning::
    As of version 0.45.x the internal implementation for the list datatype in
    Numba is changing. Until recently, only a single implementation of the list
    datatype was available, the so-called *reflected-list* (see below).
    However, it was scheduled for deprecation from version 0.44.0 onwards due
    to its limitations. As of version 0.45.0 a new implementation, the
    so-called *typed-list* (see below), is available as an experimental
    feature. For more information, please see: :ref:`deprecation`.

Creating and returning lists from JIT-compiled functions is supported,
as well as all methods and operations.  Lists must be strictly homogeneous:
Numba will reject any list containing objects of different types, even if
the types are compatible (for example, ``[1, 2.5]`` is rejected as it
contains a :class:`int` and a :class:`float`).

For example, to create a list of arrays::

  In [1]: from numba import njit

  In [2]: import numpy as np

  In [3]: @njit
    ...: def foo(x):
    ...:     lst = []
    ...:     for i in range(x):
    ...:         lst.append(np.arange(i))
    ...:     return lst
    ...:

  In [4]: foo(4)
  Out[4]: [array([], dtype=int64), array([0]), array([0, 1]), array([0, 1, 2])]


.. _feature-reflected-list:

List Reflection
'''''''''''''''

In nopython mode, Numba does not operate on Python objects.  ``list`` are
compiled into an internal representation.  Any ``list`` arguments must be
converted into this representation on the way in to nopython mode and their
contained elements must be restored in the original Python objects via a
process called :term:`reflection`.  Reflection is required to maintain the same
semantics as found in regular Python code.  However, the reflection process
can be expensive for large lists and it is not supported for lists that contain
reflected data types.  Users cannot use list-of-list as an argument because
of this limitation.

.. note::
   When passing a list into a JIT-compiled function, any modifications
   made to the list will not be visible to the Python interpreter until
   the function returns.  (A limitation of the reflection process.)

.. warning::
   List sorting currently uses a quicksort algorithm, which has different
   performance characterics than the algorithm used by Python.


.. _feature-typed-list:

Typed List
''''''''''

.. note::
  ``numba.typed.List`` is an experimental feature, if you encounter any bugs in
  functionality or suffer from unexpectedly bad performance, please report
  this, ideally by opening an issue on the Numba issue tracker.

As of version 0.45.0 a new implementation of the list data type is available,
the so-called *typed-list*. This is compiled library backed, type-homogeneous
list data type that is an improvement over the *reflected-list* mentioned
above.  Additionally, lists can now be arbitrarily nested. Since the
implementation is considered experimental, you will need to import it
explicitly from the `numba.typed` module::

    In [1]: from numba.typed import List

    In [2]: from numba import njit

    In [3]: @njit
    ...: def foo(l):
    ...:     l.append(23)
    ...:     return l
    ...:

    In [4]: mylist = List()

    In [5]: mylist.append(1)

    In [6]: foo(mylist)
    Out[6]: ListType[int64]([1, 23])


.. note::
    As the typed-list stabilizes it will fully replace the reflected-list and the
    constructors `[]` and `list()` will create a typed-list instead of a
    reflected one.


Here's an example using ``List()`` to create ``numba.typed.List`` inside a
jit-compiled function and letting the compiler infer the item type:

.. literalinclude:: ../../../examples/typed_list_usage.py
   :language: python
   :caption: from ``ex_inferred_list_jit`` of ``examples/typed_list_usage.py``
   :start-after: magictoken.ex_inferred_list_jit.begin
   :end-before: magictoken.ex_inferred_list_jit.end
   :dedent: 4
   :linenos:

Here's an example of using ``List()`` to create a ``numba.typed.List`` outside of
a jit-compiled function and then using it as an argument to a jit-compiled
function:

.. literalinclude:: ../../../examples/typed_list_usage.py
   :language: python
   :caption: from ``ex_inferred_list`` of ``examples/typed_list_usage.py``
   :start-after: magictoken.ex_inferred_list.begin
   :end-before: magictoken.ex_inferred_list.end
   :dedent: 4
   :linenos:

Finally, here's an example of using a nested `List()`:

.. literalinclude:: ../../../examples/typed_list_usage.py
   :language: python
   :caption: from ``ex_nested_list`` of ``examples/typed_list_usage.py``
   :start-after: magictoken.ex_nested_list.begin
   :end-before: magictoken.ex_nested_list.end
   :dedent: 4
   :linenos:

.. _pysupported-comprehension:

List comprehension
''''''''''''''''''

Numba supports list comprehension.  For example::


  In [1]: from numba import njit

  In [2]: @njit
    ...: def foo(x):
    ...:     return [[i for i in range(n)] for n in range(x)]
    ...:

  In [3]: foo(3)
  Out[3]: [[], [0], [0, 1]]


.. note::
  Prior to version 0.39.0, Numba did not support the creation of nested lists.


Numba also supports "array comprehension" that is a list comprehension
followed immediately by a call to :func:`numpy.array`. The following
is an example that produces a 2D Numpy array::

    from numba import jit
    import numpy as np

    @jit(nopython=True)
    def f(n):
      return np.array([ [ x * y for x in range(n) ] for y in range(n) ])

In this case, Numba is able to optimize the program to allocate and
initialize the result array directly without allocating intermediate
list objects.  Therefore, the nesting of list comprehension here is
not a problem since a multi-dimensional array is being created here
instead of a nested list.

Additionally, Numba supports parallel array comprehension when combined
with the :ref:`parallel_jit_option` option on CPUs.

set
---

All methods and operations on sets are supported in JIT-compiled functions.

Sets must be strictly homogeneous: Numba will reject any set containing
objects of different types, even if the types are compatible (for example,
``{1, 2.5}`` is rejected as it contains a :class:`int` and a :class:`float`).

.. note::
   When passing a set into a JIT-compiled function, any modifications
   made to the set will not be visible to the Python interpreter until
   the function returns.

.. _feature-typed-dict:

Typed Dict
''''''''''

.. warning::
  ``numba.typed.Dict`` is an experimental feature.  The API may change
  in the future releases.

.. note::
  ``dict()`` was not supported in versions prior to 0.44.  Currently, calling
  ``dict()`` translates to calling ``numba.typed.Dict()``.

Numba only supports the use of ``dict()`` without any arguments.  Such use is
semantically equivalent to ``{}`` and ``numba.typed.Dict()``.  It will create
an instance of ``numba.typed.Dict`` where the key-value types will be later
inferred by usage.

Numba does not fully support the Python ``dict`` because it is an untyped
container that can have any Python types as members. To generate efficient
machine code, Numba needs the keys and the values of the dictionary to have
fixed types, declared in advance. To achieve this, Numba has a typed dictionary,
``numba.typed.Dict``, for which the type-inference mechanism must be able to
infer the key-value types by use, or the user must explicitly declare the
key-value type using the ``Dict.empty()`` constructor method.
This typed dictionary has the same API as the Python ``dict``,  it implements
the ``collections.MutableMapping`` interface and is usable in both interpreted
Python code and JIT-compiled Numba functions.
Because the typed dictionary stores keys and values in Numba's native,
unboxed data layout, passing a Numba dictionary into nopython mode has very low
overhead. However, this means that using a typed dictionary from the Python
interpreter is slower than a regular dictionary because Numba has to box and
unbox key and value objects when getting or setting items.

An important difference of the typed dictionary in comparison to Python's
``dict`` is that **implicit casting** occurs when a key or value is stored.
As a result the *setitem* operation may fail should the type-casting fail.

It should be noted that the Numba typed dictionary is implemented using the same
algorithm as the CPython 3.7 dictionary. As a consequence, the typed dictionary
is ordered and has the same collision resolution as the CPython implementation.

Further to the above in relation to type specification, there are limitations
placed on the types that can be used as keys and/or values in the typed
dictionary, most notably the Numba ``Set`` and ``List`` types are currently
unsupported. Acceptable key/value types include but are not limited to: unicode
strings, arrays (value only), scalars, tuples. It is expected that these 
limitations will be relaxed as Numba continues to improve.

Here's an example of using ``dict()`` and ``{}`` to create ``numba.typed.Dict``
instances and letting the compiler infer the key-value types:

.. literalinclude:: ../../../examples/dict_usage.py
   :language: python
   :caption: from ``ex_inferred_dict_njit`` of ``examples/dict_usage.py``
   :start-after: magictoken.ex_inferred_dict_njit.begin
   :end-before: magictoken.ex_inferred_dict_njit.end
   :dedent: 4
   :linenos:

Here's an example of creating a ``numba.typed.Dict`` instance from interpreted
code and using the dictionary in jit code:

.. literalinclude:: ../../../examples/dict_usage.py
   :language: python
   :caption: from ``ex_typed_dict_from_cpython`` of ``examples/dict_usage.py``
   :start-after: magictoken.ex_typed_dict_from_cpython.begin
   :end-before: magictoken.ex_typed_dict_from_cpython.end
   :dedent: 4
   :linenos:

Here's an example of creating a ``numba.typed.Dict`` instance from jit code and
using the dictionary in interpreted code:

.. literalinclude:: ../../../examples/dict_usage.py
   :language: python
   :caption: from ``ex_typed_dict_njit`` of ``examples/dict_usage.py``
   :start-after: magictoken.ex_typed_dict_njit.begin
   :end-before: magictoken.ex_typed_dict_njit.end
   :dedent: 4
   :linenos:

It should be noted that ``numba.typed.Dict`` is not thread-safe.
Specifically, functions which modify a dictionary from multiple
threads will potentially corrupt memory, causing a
range of possible failures. However, the dictionary can be safely read from
multiple threads as long as the contents of the dictionary do not
change during the parallel access.

None
----

The None value is supported for identity testing (when using an
:class:`~numba.optional` type).


bytes, bytearray, memoryview
----------------------------

The :class:`bytearray` type and, on Python 3, the :class:`bytes` type
support indexing, iteration and retrieving the len().

The :class:`memoryview` type supports indexing, slicing, iteration,
retrieving the len(), and also the following attributes:

* :attr:`~memoryview.contiguous`
* :attr:`~memoryview.c_contiguous`
* :attr:`~memoryview.f_contiguous`
* :attr:`~memoryview.itemsize`
* :attr:`~memoryview.nbytes`
* :attr:`~memoryview.ndim`
* :attr:`~memoryview.readonly`
* :attr:`~memoryview.shape`
* :attr:`~memoryview.strides`


Built-in functions
==================

The following built-in functions are supported:

* :func:`abs`
* :class:`bool`
* :class:`complex`
* :func:`divmod`
* :func:`enumerate`
* :class:`float`
* :func:`hash` (see :ref:`pysupported-hashing` below)
* :class:`int`: only the one-argument form
* :func:`iter`: only the one-argument form
* :func:`len`
* :func:`min`
* :func:`max`
* :func:`next`: only the one-argument form
* :func:`print`: only numbers and strings; no ``file`` or ``sep`` argument
* :class:`range`: semantics are similar to those of Python 3 even in Python 2:
  a range object is returned instead of an array of values. The only permitted
  use of range is as a callable function (cannot pass range as an argument to a
  jitted function or return a range from a jitted function).
* :func:`round`
* :func:`sorted`: the ``key`` argument is not supported
* :func:`type`: only the one-argument form, and only on some types
  (e.g. numbers and named tuples)
* :func:`zip`

.. _pysupported-hashing:

Hashing
-------

The :func:`hash` built-in is supported and produces hash values for all
supported hashable types with the following Python version specific behavior:

Under Python 3, hash values computed by Numba will exactly match those computed
in CPython under the condition that the :attr:`sys.hash_info.algorithm` is
``siphash24`` (default).

Under Python 2, hash values computed by Numba will follow the behavior
described for Python 3 with the :attr:`sys.hash_info.algorithm` emulated as
``siphash24``. No attempt is made to replicate Python 2 hashing behavior.

The ``PYTHONHASHSEED`` environment variable influences the hashing behavior in
precisely the manner described in the CPython documentation.


Standard library modules
========================

``array``
---------

Limited support for the :class:`array.array` type is provided through
the buffer protocol.  Indexing, iteration and taking the len() is supported.
All type codes are supported except for ``"u"``.

``cmath``
---------

The following functions from the :mod:`cmath` module are supported:

* :func:`cmath.acos`
* :func:`cmath.acosh`
* :func:`cmath.asin`
* :func:`cmath.asinh`
* :func:`cmath.atan`
* :func:`cmath.atanh`
* :func:`cmath.cos`
* :func:`cmath.cosh`
* :func:`cmath.exp`
* :func:`cmath.isfinite`
* :func:`cmath.isinf`
* :func:`cmath.isnan`
* :func:`cmath.log`
* :func:`cmath.log10`
* :func:`cmath.phase`
* :func:`cmath.polar`
* :func:`cmath.rect`
* :func:`cmath.sin`
* :func:`cmath.sinh`
* :func:`cmath.sqrt`
* :func:`cmath.tan`
* :func:`cmath.tanh`

``collections``
---------------

Named tuple classes, as returned by :func:`collections.namedtuple`, are
supported in the same way regular tuples are supported.  Attribute access
and named parameters in the constructor are also supported.

Creating a named tuple class inside Numba code is *not* supported; the class
must be created at the global level.

.. _ctypes-support:

``ctypes``
----------

Numba is able to call ctypes-declared functions with the following argument
and return types:

* :class:`ctypes.c_int8`
* :class:`ctypes.c_int16`
* :class:`ctypes.c_int32`
* :class:`ctypes.c_int64`
* :class:`ctypes.c_uint8`
* :class:`ctypes.c_uint16`
* :class:`ctypes.c_uint32`
* :class:`ctypes.c_uint64`
* :class:`ctypes.c_float`
* :class:`ctypes.c_double`
* :class:`ctypes.c_void_p`

``enum``
--------

Both :class:`enum.Enum` and :class:`enum.IntEnum` subclasses are supported.

``math``
--------

The following functions from the :mod:`math` module are supported:

* :func:`math.acos`
* :func:`math.acosh`
* :func:`math.asin`
* :func:`math.asinh`
* :func:`math.atan`
* :func:`math.atan2`
* :func:`math.atanh`
* :func:`math.ceil`
* :func:`math.copysign`
* :func:`math.cos`
* :func:`math.cosh`
* :func:`math.degrees`
* :func:`math.erf`
* :func:`math.erfc`
* :func:`math.exp`
* :func:`math.expm1`
* :func:`math.fabs`
* :func:`math.floor`
* :func:`math.frexp`
* :func:`math.gamma`
* :func:`math.hypot`
* :func:`math.isfinite`
* :func:`math.isinf`
* :func:`math.isnan`
* :func:`math.ldexp`
* :func:`math.lgamma`
* :func:`math.log`
* :func:`math.log10`
* :func:`math.log1p`
* :func:`math.pow`
* :func:`math.radians`
* :func:`math.sin`
* :func:`math.sinh`
* :func:`math.sqrt`
* :func:`math.tan`
* :func:`math.tanh`
* :func:`math.trunc`

``operator``
------------

The following functions from the :mod:`operator` module are supported:

* :func:`operator.add`
* :func:`operator.and_`
* :func:`operator.div` (Python 2 only)
* :func:`operator.eq`
* :func:`operator.floordiv`
* :func:`operator.ge`
* :func:`operator.gt`
* :func:`operator.iadd`
* :func:`operator.iand`
* :func:`operator.idiv` (Python 2 only)
* :func:`operator.ifloordiv`
* :func:`operator.ilshift`
* :func:`operator.imatmul` (Python 3.5 and above)
* :func:`operator.imod`
* :func:`operator.imul`
* :func:`operator.invert`
* :func:`operator.ior`
* :func:`operator.ipow`
* :func:`operator.irshift`
* :func:`operator.isub`
* :func:`operator.itruediv`
* :func:`operator.ixor`
* :func:`operator.le`
* :func:`operator.lshift`
* :func:`operator.lt`
* :func:`operator.matmul` (Python 3.5 and above)
* :func:`operator.mod`
* :func:`operator.mul`
* :func:`operator.ne`
* :func:`operator.neg`
* :func:`operator.not_`
* :func:`operator.or_`
* :func:`operator.pos`
* :func:`operator.pow`
* :func:`operator.rshift`
* :func:`operator.sub`
* :func:`operator.truediv`
* :func:`operator.xor`

``functools``
-------------

The :func:`functools.reduce` function is supported but the `initializer`
argument is required.

.. _pysupported-random:

``random``
----------

Numba supports top-level functions from the :mod:`random` module, but does
not allow you to create individual Random instances.  A Mersenne-Twister
generator is used, with a dedicated internal state.  It is initialized at
startup with entropy drawn from the operating system.

* :func:`random.betavariate`
* :func:`random.expovariate`
* :func:`random.gammavariate`
* :func:`random.gauss`
* :func:`random.getrandbits`: number of bits must not be greater than 64
* :func:`random.lognormvariate`
* :func:`random.normalvariate`
* :func:`random.paretovariate`
* :func:`random.randint`
* :func:`random.random`
* :func:`random.randrange`
* :func:`random.seed`: with an integer argument only
* :func:`random.shuffle`: the sequence argument must be a one-dimension
  Numpy array or buffer-providing object (such as a :class:`bytearray`
  or :class:`array.array`); the second (optional) argument is not supported
* :func:`random.uniform`
* :func:`random.triangular`
* :func:`random.vonmisesvariate`
* :func:`random.weibullvariate`

.. note::
   Calling :func:`random.seed` from non-Numba code (or from :term:`object mode`
   code) will seed the Python random generator, not the Numba random generator.

.. note::
   Since version 0.28.0, the generator is thread-safe and fork-safe.  Each
   thread and each process will produce independent streams of random numbers.

.. seealso::
   Numba also supports most additional distributions from the :ref:`Numpy
   random module <numpy-random>`.

``heapq``
------------

The following functions from the :mod:`heapq` module are supported:

* :func:`heapq.heapify`
* :func:`heapq.heappop`
* :func:`heapq.heappush`
* :func:`heapq.heappushpop`
* :func:`heapq.heapreplace`
* :func:`heapq.nlargest` : first two arguments only
* :func:`heapq.nsmallest` : first two arguments only

Note: the heap must be seeded with at least one value to allow its type to be
inferred; heap items are assumed to be homogeneous in type.


Third-party modules
===================

.. I put this here as there's only one module (apart from Numpy), otherwise
   it should be a separate page.

.. _cffi-support:

``cffi``
--------

Similarly to ctypes, Numba is able to call into `cffi`_-declared external
functions, using the following C types and any derived pointer types:

* :c:type:`char`
* :c:type:`short`
* :c:type:`int`
* :c:type:`long`
* :c:type:`long long`
* :c:type:`unsigned char`
* :c:type:`unsigned short`
* :c:type:`unsigned int`
* :c:type:`unsigned long`
* :c:type:`unsigned long long`
* :c:type:`int8_t`
* :c:type:`uint8_t`
* :c:type:`int16_t`
* :c:type:`uint16_t`
* :c:type:`int32_t`
* :c:type:`uint32_t`
* :c:type:`int64_t`
* :c:type:`uint64_t`
* :c:type:`float`
* :c:type:`double`
* :c:type:`ssize_t`
* :c:type:`size_t`
* :c:type:`void`

The ``from_buffer()`` method of ``cffi.FFI`` and ``CompiledFFI`` objects is
supported for passing Numpy arrays and other buffer-like objects.  Only
*contiguous* arguments are accepted.  The argument to ``from_buffer()``
is converted to a raw pointer of the appropriate C type (for example a
``double *`` for a ``float64`` array).

Additional type mappings for the conversion from a buffer to the appropriate C
type may be registered with Numba. This may include struct types, though it is
only permitted to call functions that accept pointers to structs - passing a
struct by value is unsupported. For registering a mapping, use:

.. function:: numba.cffi_support.register_type(cffi_type, numba_type)

Out-of-line cffi modules must be registered with Numba prior to the use of any
of their functions from within Numba-compiled functions:

.. function:: numba.cffi_support.register_module(mod)

   Register the cffi out-of-line module ``mod`` with Numba.

Inline cffi modules require no registration.

.. _cffi: https://cffi.readthedocs.org/
