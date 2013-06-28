Numba Language Specification
============================

This document attempts to specify the Python subset supported by the numba
compiler, and attempts to clarify which constructs are supported natively
without help of the object layer.

Native Types
------------

- bool
- char
- int
- float
- complex
- string [2]
- arrays [1], [2] and array expressions
- extension types [1]
- numba functions
- Ctypes/CFFI functions
- pointers
- structs

.. NOTE:: [1] with reference counting
.. NOTE:: [2] indexing, slicing and len()

Boxed Types
-----------

The following types are currently boxed in PyObjects:

- unicode
- list
- tuple
- set
- dict
- object
- frozenset
- buffer
- bytearray
- bytes
- memoryview

All operations on these types go through the object layer.

Values
------

Tuple unpacking is supported for:

    - arrays of known size, e.g. ``m, n = array.shape``
    - syntactic assignment, e.g. ``x, y = a, b``

Anything else goes through the object layer.

Control Flow
------------

Supported:

- If
- If/Else
- If/ElseIf/Else
- For
- For/Else
- While
- While/Else
- Return
- Raise

Not Supported:

- Generators
- Try
- Try/Finally
- Try/Except
- Try/Except/Finally

Introspection
-------------

Runtime introspection with ``type``, ``isinstance``,
``issubclass``, ``id``, ``dir``,
``callable``, ``getattr``, ``hash``, ``hasattr``, ``super`` and
``vars`` are supported only through the object layer.

``globals``, ``locals`` are not supported.

Length
------

The implementation of the ``len`` function is polymorphic and
container specific.

::

    len :: [a] -> int

Destruction
-----------

Variable and element destruction is not supported. The ``del``
operator is not part of the syntax and ``delattr` is not
supported.

Metaprogramming
---------------

``compile``, ``eval`` and ``exec``, ``execfile`` are not supported

Pass
----

Pass is ignored.

System IO
---------

``file``, ``open`` and ``quit``, ``raw_input``, ``reload``,
``help`` and ``input`` are are supported only through the object layer.

``print`` is supported through the object layer (default) or through
``printf`` (nopython mode).

Formatting
----------

String formatting is supported through the object layer.

Iterators
---------

Generator definitions are not supported. Generator and iterator
iteration is supported through the object layer.

Range iterators are syntactic sugar for looping constructs. Custom
iterators are not supported. The ``iter`` and ``next`` functions
are are supported only through the object layer.

::

    for i in xrange(start, stop, step):
        foo()

Is lowered into some equivalent low-level looping construct that
roughly corresponds to the following C code:

.. code-block:: c

    for (i = start; i < stop; i += step) {
        foo();
    }

The value of ``i`` after the loop block follows the Python
semantics and is set to the last value in the iterator instead of
the C semantics.

``xrange`` and `range`` are lowered into the same constructs.

``enumerate`` is supported through the object layer.

Comprehensions
--------------

List comprehensions are rewritten into equivalent loops with list appending.
Generator comprehensions are not supported.

Builtins
--------

* abs            -  Supported
* all            -  object layer
* any            -  object layer
* apply          -  object layer
* basestring     -  object layer
* bin            -  object layer
* bool           -  Supported
* buffer         -  object layer
* bytearray      -  object layer
* bytes          -  object layer
* callable       -  object layer
* chr            -  object layer
* classmethod    -  object layer
* cmp            -  object layer
* coerce         -  object layer
* compile        -  object layer
* complex        -  Supported
* delattr        -  object layer
* dict           -  object layer
* dir            -  object layer
* divmod         -  object layer
* enumerate      -  object layer
* eval           -  object layer
* execfile       -  object layer
* exit           -  object layer
* file           -  object layer
* filter         -  Supported
* float          -  Supported
* format         -  object layer
* frozenset      -  object layer
* getattr        -  object layer
* globals        -  object layer
* hasattr        -  object layer
* hash           -  object layer
* help           -  object layer
* hex            -  object layer
* id             -  object layer
* input          -  object layer
* int            -  Supported
* intern         -  object layer
* isinstance     -  object layer
* issubclass     -  object layer
* iter           -  object layer
* len            -  Supported
* list           -  object layer
* locals         -  object layer
* long           -  Supported
* map            -  object layer
* max            -  object layer
* memoryview     -  object layer
* min            -  Supported
* next           -  object layer
* object         -  object layer
* oct            -  object layer
* open           -  object layer
* ord            -  object layer
* pow            -  Supported
* print          -  Supported
* property       -  object layer
* quit           -  object layer
* range          -  Supported
* raw_input      -  object layer
* reduce         -  object layer
* reload         -  object layer
* repr           -  object layer
* reversed       -  object layer
* round          -  Supported
* set            -  object layer
* setattr        -  object layer
* slice          -  object layer
* sorted         -  object layer
* staticmethod   -  object layer
* str            -  Supported
* sum            -  object layer
* super          -  object layer
* tuple          -  object layer
* type           -  object layer
* unichr         -  object layer
* unicode        -  object layer
* vars           -  object layer
* xrange         -  Supported
* zip            -  object layer

Slice
-----

Named slicing is not supported. Slice types are supported only through the
object layer. Slicing as an indexing operation is supported.

::

    a = slice(0, 1, 2)

Classes
-------

Classes are supported through extension types.

Casts
-----

::

    int :: a -> int
    bool :: a -> bool
    complex :: a -> bool

The coerce function is not supported.

The ``str``, ``list`` and ``tuple`` casts are not supported.

Characters
----------

The ``chr``, ``ord`` are supported for the integer and character types.
``unichr``, ``hex``, ``bin``, ``oct`` functions are supported through the
object layer.

Closures
--------

Nested functions and closures are supported. Construction goes through
the object layer. Calling from numba does not.

The ``nonlocal`` keyword is not supported.

Globals
-------

Global variables are not supported and resolved as constants. The ``global``
keyword is not supported.

Arguments
---------

Variadic and keyword arguments are not supported.

Assertions
----------

Assertions are not supported.

Operators
---------

- And
- Or
- Add
- Sub
- Mult
- Div
- Mod
- Pow
- LShift
- RShift
- BitOr
- BitXor
- BitAnd
- FloorDiv
- Invert
- Not
- UAdd
- USub
- Eq
- NotEq
- Lt
- LtE
- Gt
- GtE

Comparison operator chaining is supported and is desugared into
boolean conjunctions of the comparison operators::

    (x > y > z)

::

    (x > y) and (y > z)

Not supported:

- Is
- IsNot
- In
- NotIn

Division
--------

Division follows the Python semantics for distinction between
``floordiv`` and ``truediv`` but operates over unboxed types
with no error checking.

Math Functions
--------------

- abs
- pow
- round

Floating Point Math
-------------------

* acos
* acosh
* asin
* asinh
* atan
* atan2
* atanh
* ceil
* cos
* cosh
* degrees
* erf
* erfc
* exp
* expm1
* exp2
* fabs
* floor
* fmod
* hypot
* log
* logaddexp
* logaddexp2
* log10
* log1p
* modf
* pow
* rint
* sin
* sinh
* sqrt
* tan
* tanh
* trunc

Constants such as ``math.e`` and ``math.pi`` are resolved as constants.

Complex Math
-------------------

* abs
* acos
* acosh
* asin
* asinh
* atan
* atanh
* cos
* cosh
* exp
* expm1
* exp2
* log
* log10
* log1p
* sin
* sinh
* sqrt
* tan
* tanh
