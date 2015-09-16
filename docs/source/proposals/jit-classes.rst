===================
NBEP 2: JIT Classes
===================

:Author: Siu Kwan Lam
:Date: Sept 2015
:Status: Draft

Introduction
============

Numba does not yet support user-defined classes.
Classes provide useful abstraction and promote modularity when used
right.  In the simplest sense, a class specifies the set of data and
operations as attributes and methods, respectively.
A class instance is an instantiation of that class.
This proposal will focus on supporting this simple usecase of classes--with
just attributes and methods.  Other features, such as class methods, static
methods, and inheritance are deferred to another proposal, but we believe
these features can be easily implemented given the foundation described here.


Proposal: jit-classes
=====================

A JIT-classes is more restricted than a Python class.
We will focus on the following operations on a class and its instance:

* Instantiation: create an instance of a class using the class object as the
  constructor: ``cls(*args, **kwargs)``
* Destruction: remove resources allocated during instantiation and release
  all references to other objects.
* Attribute access: loading and storing attributes using ``instance.attr``
  syntax.
* Method access: loading methods using ``instance.method`` syntax.

With these operations, a class object (not the instance) does not need to be
materialize. Using the class object as a constructor is fully resolved (a
runtime implementation is picked) during the typing phase in the compiler.
This means **a class object will not be first class**.  On the other hand,
implementating a first-class class object will require an
"interface" type, or the type of class.

The instantiation of a class will allocate resources for storing the data
attributes.  This is described in the "Storage model" section.  Methods are
never stored in the instance.  They are information attached to the class.
Since a class object only exists in the type domain, the methods will also be
fully resolved at the typing phase.  Again, numba do not have first-class
function value and each function type maps uniquely to each function
implementation (this needs to be changed to support function value as argument).

A class instance can contain other NRT reference-counted object as attributes.
To properly clean up an instance, a desctruction is called when the reference
count of the instance is dropped to zero.  This is described in the
"Reference count and descructor" section.

Storage model
~~~~~~~~~~~~~

For compatibility with C, attributes are stored in a simple plain-old-data
structure.  Each attribute are stored in a user-defined order in a padded
(for proper alignment), contiguous memory region. An instance that contains
three fields of int32, float32, complex64 will be compatible with the following
C structure::

    struct {
        int32     field0;
        float32   field1;
        complex64 field2;
    };

This will also be comptabile with an aligned numpy structure dtype.


Methods
~~~~~~~

Methods are regular function that can be bounded to an instance.
They can be compiled as regular function by numba.
The operation ``getattr(instance, name)`` (getting an attribute ``name`` from
``instance``) performs the bounding if the attribute is a method.


The special ``__init__`` method is also handled like regular functions.


``__del__`` is not supported at this time.


Reference count and destructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An instance of jit-class is reference-counted by NRT. Since it may contain
other NRT tracked object, it must call a destructor when its reference count
dropped to zero.  The destructor will decrement the reference count of all
attributes by one.

At this time, there is no support for user defined ``__del__`` method.

Cyclic reference is not handled at this time.  It will cause memory leak.

Type inference
~~~~~~~~~~~~~~

So far we have not described the type of the attributes or the methods.
Type information is necessary to materailize the instance (e.g. allocate the
storage).  The simplest way is to let user provide the type of each attributes
as well as the ordering; for instance::

    dct = OrderedDict()
    dct['x'] = int32
    dct['y'] = float32

Allowing user to supply an ordered dictionary will provide the name, ordering
and types of the attributes.  However, this statically typed semantic is not as
flexible as the Python semantic which behaves like a generic (or C++ template)
class.

Inferring the type of attributes is difficult.  In a previous attempt to
implement JIT classes, the ``__init__`` method is specialized to capture
the type stored into the attributes.  Since the method can contain arbitrary
logic, the problem can become a dependent typing problem if types are assigned
conditionally depending on the value. (Very few languages implement dependent
typing and those that does are mostly theorem provers.)

Alternatively, we can let user provide the typing logic.  For example, user
can supply a function that runs in the type domain::

    def get_type(x, y):
        dct = OrderedDict()
        if x == int32:
            dct['x'] = float32
        else:
            raise TypeError
        dct['y'] = y
        return dct

    class Foo:
        def __init__(self, x, y):
            self.a = x
            self.b = y

The ``get_type`` takes the same arguments as the ``__init__`` of Foo but in
the type domain (e.g. ``get_type(typeof(x), typeof(y)``).  In the function,
the type of each attributes is decided.  The ``get_type`` function requires
``x`` to be of ``int32`` and uses ``float32`` for attribute ``a``.  Attribute
``b`` can be of any type and it is the same as the type of ``y``.  This scheme
will support generic classes without the need of solving a difficult type
inference problem.


Immutable class: jit-struct
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jit-classes are flexible and they can be too flexible.  The flexibility
requires runtime assistance that is not available or inefficient in more
some hardware, such as the GPGPU targets.  One way to limit the flexibility
is to enforce immutability. We call this a *jit-struct*.  Python has immutable
containers such as the tuple.  A jit-struct will behave like a jit-class with
a restriction that disallow mutating any attributes after instantiation.
In other words, mutation of attributes is only allowed inside ``__init__``.

This can be easily done by changing the type of ``self`` argument passed to
the ``__init__``.  A jit-struct instance is a pass-by-value structure with
the exception during instantiation that a pass-by-reference structure is passed
to ``__init__``.

User API
~~~~~~~~


Typing function as external function
------------------------------------

.. code-block:: python

    def spec(x, y):
        dct = OrderedDict()
        dct['x'] = x
        dct['y'] = y
        return dct

    @jit(spec, immutable=False)
    class Vec(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def add(self, dx, dy):
            self.x += dx
            self.y += dy

Typing function as static method
--------------------------------

.. code-block:: python

    def spec(x, y):
        dct = OrderedDict()
        dct['x'] = x
        dct['y'] = y
        return dct

    @jit(immutable=False)
    class Vec(object):
        @staticmethod
        def __type_inference__(x, y):
            dct = OrderedDict()
            dct['x'] = x
            dct['y'] = y
            return dct

        def __init__(self, x, y):
            self.x = x
            self.y = y

        def add(self, dx, dy):
            self.x += dx
            self.y += dy



