********************
Python Functionality
********************

.. _extclasses:

Extension Types (Classes)
=========================
Numba supports classes similar to Python classes and extension types.
Classes, like functions, come in two flavours:

    * jit
    * autojit

Below we shall discuss each individually.

jit classes
------------
Methods of jit classes have static signatures, where all arguments
except ``self`` or ``cls`` are specified, e.g.::

    @jit
    class MyExtension(object):
        @void(double)
        def __init__(self, value):
            self.value = value

        @classmethod
        @object_(double)
        def from_value(cls, value):
            return cls(value)

    obj = MyExtension(10.0)

One can refer to the extension type as a type by accessing its ``exttype``
attribute::

    @jit(MyExtension.exttype(double))
    def create_ext(arg):
        return MyExtension(arg)

It is not yet possible to refer to the extension type in the class body or
methods of that extension type.

Things that work:

    * overriding Numba methods in Numba (all methods are virtual)
    * inheritance
    * instance attributes
    * subclassing in python and calling overridden methods in Python
    * arbitrary new attributes on extension classes and objects
    * weakrefs to extension objects
    * class- and staticmethods

Things that do NOT (yet) work:

    * overriding methods in Python and calling the method from Numba
    * multiple inheritance of Numba classes

        - (multiple inheritance with Python classes should work)

    * subclassing variable sized objects like 'str' or 'tuple'

An more detailed example of extension classes and their capabilities
can be found here: :ref:`classes`.

.. .. literalinclude:: /../../examples/numbaclasses.py

autojit classes
---------------
Autojit classes specialize on their arguments when they are called. Autojit
classes can have static method signatures analogous to methods of jit classes.
They can also have autojit methods, which are methods that specialize on
their argument types when called::

    @autojit
    class MyAutojitExtension(object):
        def __init__(self, value):
            self.value = value

        @int32(int32)
        def method1(self, arg):
            return arg + self.value

        def method2(self, arg):
            return arg + self.value

We have an autojit class that specializes to one argument (``value``) when called.
``method1`` specializes to instance attribute ``value``, but non to its argument.
``method2`` on the other hand specializes to both the instance attribute and
its argument value:

.. code-block:: pycon

    >>> obj = MyAutojitExtension(10)
    >>> obj.method1(2)
    12
    >>> obj.method1(2.0)
    12
    >>> obj.method2(2)
    12
    >>> obj.method2(2.0)
    12.0

Autojit classes mostly work like Python classes. We can still use static-,
class- and unbound methods:

.. code-block:: pycon

    >>> MyAutojitExtension.method2(obj, 2+2j)
    (12+2j)

One may retrieve specialized class instances through class indexing::

    >>> print MyAutojitExtension[{'value': int_}].exttype
    <AutojitExtension MyAutojitExtension({'value': int})>

And inspect which specializations are available::

    >>> MyAutojitExtension.specializations
    [<class 'MyAutojitExtension'>]

Class specializations may go away in the future, where specialized methods
are bound on instances (to be callable from python), and attributes stored
seperately on the heap. Specialized classes are always subclasses of the
unspecialized class::

    >>> isinstance(obj, MyAutojitExtension)
    True

.. _closures:

Closures
========

Numba supports closures (nested functions), and keeps the variable scopes
alive for the lifetime of the closures.
Variables that are closed over by the closures (``cell variables``) have
one consistent type throughout the entirety of the function. This means
differently typed variables can only be assigned if they are unifyable,
such as for instance ints and floats::

    @autojit
    def outer(arg1, arg2):
        arg1 = 0
        arg1 = 0.0      # This is fine
        arg1 = "hello"  # ERROR! arg1 must have a single type

        arg2 = 0
        arg2 = "hello"  # FINE! Not a cell variable

        def inner():
            print arg1

        return inner

Calling an inner function directly in the body of ``outer`` will result in
a direct, native call of the closure. In the future it is likely that passing
around the closure will still result in a native call in other places.

Like Python closures, closures can be arbitrarily nested, and follow the same
scoping rules.

.. _containers:

Typed Containers
================
Numba ships implementations of various typed containers, which allow fast
execution and memory-efficient storage.

We hope to support the following container types:

    * list, tuple
    * dict, ordereddict
    * set, orderedset
    * queues, channels
    * fixedlist (fixed number of element and each element can have different type)
    * <your idea here>

There are many more data structure that can be implemented, but future releases
of numba will make it easier (nearly trivial) for people to implement those
data structure themselves while supporting full data polymorphism.

Currently implemented:

    * typedlist
    * typedtuple

These data structures work exactly like their python equivalents, but take a
first parameter which specifies the element type::

    >>> numba.typedlist(int32, range(10))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> numba.typedlist(float32, range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    >>> tlist = numba.typedlist(int32)
    >>> tlist
    []
    >>> tlist.extend([3, 2, 1, 3])
    >>> tlist
    [3, 1, 2, 3]
    >>> tlist.count(3)
    2L
    >>> tlist[0]
    3L
    >>> tlist.pop()
    3L
    >>> tlist.reverse()
    >>> tlist
    [1, 2, 3]

Things that are not yet implemented:

    * Methods ``remove``, ``insert``
    * Slicing

Typed containers can be used from Python or from numba code. Using them from numba code
will result in fast calls without boxing and unboxing.

