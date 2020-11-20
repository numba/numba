.. _jitclass:

===========================================
Compiling Python classes with ``@jitclass``
===========================================

.. note::

  This is a early version of jitclass support. Not all compiling features are
  exposed or implemented, yet.


Numba supports code generation for classes via the :func:`numba.jitclass`
decorator.  A class can be marked for optimization using this decorator along
with a specification of the types of each field.  We call the resulting class
object a *jitclass*.  All methods of a jitclass are compiled into nopython
functions.  The data of a jitclass instance is allocated on the heap as a
C-compatible structure so that any compiled functions can have direct access
to the underlying data, bypassing the interpreter.


Basic usage
===========

Here's an example of a jitclass:

.. literalinclude:: ../../../numba/tests/doc_examples/test_jitclass.py
   :language: python
   :start-after: magictoken.ex_jitclass.begin
   :end-before: magictoken.ex_jitclass.end
   :dedent: 8

In the above example, a ``spec`` is provided as a list of 2-tuples.  The tuples
contain the name of the field and the Numba type of the field.  Alternatively,
user can use a dictionary (an ``OrderedDict`` preferably for stable field
ordering), which maps field names to types.

The definition of the class requires at least a ``__init__`` method for
initializing each defined fields.  Uninitialized fields contains garbage data.
Methods and properties (getters and setters only) can be defined.  They will be
automatically compiled.


Inferred class member types from type annotations with ``as_numba_type``
========================================================================

Fields of a ``jitclass`` can also be inferred from Python type annotations.

.. literalinclude:: ../../../numba/tests/doc_examples/test_jitclass.py
   :language: python
   :start-after: magictoken.ex_jitclass_type_hints.begin
   :end-before: magictoken.ex_jitclass_type_hints.end
   :dedent: 8

Any type annotations on the class will be used to extend the spec if that field
is not already present.  The Numba type corresponding to the given Python type
is inferred using ``as_numba_type``.  For example, if we have the class

.. code-block:: python

    @jitclass([("w", int32), ("y", float64[:])])
    class Foo:
        w: int
        x: float
        y: np.ndarray
        z: SomeOtherType

        def __init__(self, w: int, x: float, y: np.ndarray, z: SomeOtherType):
            ...

then the full spec used for ``Foo`` will be:

* ``"w": int32`` (specified in the ``spec``)
* ``"x": float64`` (added from type annotation)
* ``"y": array(float64, 1d, A)`` (specified in the ``spec``)
* ``"z": numba.as_numba_type(SomeOtherType)`` (added from type annotation)

Here ``SomeOtherType`` could be any supported Python type (e.g.
``bool``, ``typing.Dict[int, typing.Tuple[float, float]]``, or another
``jitclass``).

Note that only type annotations on the class will be used to infer spec
elements.  Method type annotations (e.g. those of ``__init__`` above) are
ignored.

Numba requires knowing the dtype and rank of numpy arrays, which cannot
currently be expressed with type annotations. Because of this, numpy arrays need
to be included in the ``spec`` explicitly.


Specifying ``numba.typed`` containers as class members explicitly
=================================================================

The following patterns demonstrate how to specify a ``numba.typed.Dict`` or
``numba.typed.List`` explicitly as part of the ``spec`` passed to ``jitclass``.

First, using explicit Numba types and explicit construction.

.. code-block:: python

    from numba import jitclass, types, typed

    # key and value types
    kv_ty = (types.int64, types.unicode_type)

    # A container class with:
    # * member 'd' holding a typed dictionary of int64 -> unicode string (kv_ty)
    # * member 'l' holding a typed list of float64
    @jitclass([('d', types.DictType(*kv_ty)),
               ('l', types.ListType(types.float64))])
    class ContainerHolder(object):
        def __init__(self):
            # initialize the containers
            self.d = typed.Dict.empty(*kv_ty)
            self.l = typed.List.empty_list(types.float64)

    container = ContainerHolder()
    container.d[1] = "apple"
    container.d[2] = "orange"
    container.l.append(123.)
    container.l.append(456.)
    print(container.d) # {1: apple, 2: orange}
    print(container.l) # [123.0, 456.0]

Another useful pattern is to use the ``numba.typed`` container attribute
``_numba_type_`` to find the type of a container, this can be accessed directly
from an instance of the container in the Python interpreter. The same
information can be obtained by calling :func:`numba.typeof` on the instance. For
example:

.. code-block:: python

    from numba import jitclass, typed, typeof

    d = typed.Dict()
    d[1] = "apple"
    d[2] = "orange"
    l = typed.List()
    l.append(123.)
    l.append(456.)


    @jitclass([('d', typeof(d)), ('l', typeof(l))])
    class ContainerInstHolder(object):
        def __init__(self, dict_inst, list_inst):
            self.d = dict_inst
            self.l = list_inst

    container = ContainerInstHolder(d, l)
    print(container.d) # {1: apple, 2: orange}
    print(container.l) # [123.0, 456.0]

It is worth noting that the instance of the container in a ``jitclass`` must be
initialized before use, for example, this will cause an invalid memory access
as ``self.d`` is written to without ``d`` being initialized as a ``type.Dict``
instance of the type specified.

.. code-block:: python

    from numba import jitclass, types

    dict_ty = types.DictType(types.int64, types.unicode_type)

    @jitclass([('d', dict_ty)])
    class NotInitialisingContainer(object):
        def __init__(self):
            self.d[10] = "apple" # this is invalid, `d` is not initialized

    NotInitialisingContainer() # segmentation fault/memory access violation


Support operations
==================

The following operations of jitclasses work in both the interpreter and Numba
compiled functions:

* calling the jitclass class object to construct a new instance
  (e.g. ``mybag = Bag(123)``);
* read/write access to attributes and properties (e.g. ``mybag.value``);
* calling methods (e.g. ``mybag.increment(3)``);
* calling static methods as instance attributes (e.g. ``mybag.add(1, 1)``);
* calling static methods as class attributes (e.g. ``Bag.add(1, 2)``);

Using jitclasses in Numba compiled function is more efficient.
Short methods can be inlined (at the discretion of LLVM inliner).
Attributes access are simply reading from a C structure.
Using jitclasses from the interpreter has the same overhead of calling any
Numba compiled function from the interpreter.  Arguments and return values
must be unboxed or boxed between Python objects and native representation.
Values encapsulated by a jitclass does not get boxed into Python object when
the jitclass instance is handed to the interpreter.  It is during attribute
access to the field values that they are boxed.
Calling static methods as class attributes is only supported outside of the
class definition (i.e. code cannot call ``Bag.add()`` from within another method
of ``Bag``).


Limitations
===========

* A jitclass class object is treated as a function (the constructor) inside
  a Numba compiled function.
* ``isinstance()`` only works in the interpreter.
* Manipulating jitclass instances in the interpreter is not optimized, yet.
* Support for jitclasses are available on CPU only.
  (Note: Support for GPU devices is planned for a future release.)


The decorator: ``@jitclass``
============================

.. autofunction:: numba.experimental.jitclass
