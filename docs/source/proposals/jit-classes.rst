===================
NBEP 3: JIT Classes
===================

:Author: Siu Kwan Lam
:Date: Dec 2015
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
To properly clean up an instance, a destructor is called when the reference
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
``instance``) binds the instance to the requested method at runtime.


The special ``__init__`` method is also handled like regular functions.


``__del__`` is not supported at this time.


Reference count and destructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An instance of jit-class is reference-counted by NRT. Since it may contain
other NRT tracked object, it must call a destructor when its reference count
dropped to zero.  The destructor will decrement the reference count of all
attributes by one.

At this time, there is no support for user defined ``__del__`` method.

Proper cleanup for cyclic reference is not handled at this time.
Cycles will cause memory leak.

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
flexible as the Python semantic which behaves like a generic class.

Inferring the type of attributes is difficult.  In a previous attempt to
implement JIT classes, the ``__init__`` method is specialized to capture
the type stored into the attributes.  Since the method can contain arbitrary
logic, the problem can become a dependent typing problem if types are assigned
conditionally depending on the value. (Very few languages implement dependent
typing and those that does are mostly theorem provers.)

Example: typing function using an OrderedDict
---------------------------------------------

.. code-block:: python

    spec = OrderedDict()
    spec['x'] = numba.int32
    spec['y'] = numba.float32

    @jitclass(spec)
    class Vec(object):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def add(self, dx, dy):
            self.x += dx
            self.y += dy

Example: typing function using a list of 2-tuples
-------------------------------------------------

.. code-block:: python

    spec = [('x', numba.int32),
            ('y', numba.float32)]

    @jitclass(spec)
    class Vec(object):
        ...

Creating multiple jitclasses from a single class object
-------------------------------------------------------

The `jitclass(spec)` decorator creates a new jitclass type even when applied to
the same class object and the same type specification.

.. code-block:: python

    class Vec(object):
      ...

    Vec1 = jitclass(spec)(Vec)
    Vec2 = jitclass(spec)(Vec)
    # Vec1 and Vec2 are two different jitclass types

Usage from the Interpreter
~~~~~~~~~~~~~~~~~~~~~~~~~~

When constructing a new instance of a jitclass, a "box" is created that wraps
the underlying jitclass instance from numba.  Attributes and methods are
accessible from the interpreter.  The actual implementation will be in numba
compiled code.  Any Python object is converted to its native
representation for consumption in numba.  Similarly, the returned value is
converted to its Python representation.  As a result, there may be overhead in
manipulating jitclass instances in the interpreter.  This overhead is minimal
and should be easily amortized by more efficient computation in the compiled
methods.

Support for property, staticmethod and classmethod
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The use of ``property`` is accepted for getter and setter only.  Deleter is not
supported.

The use of ``staticmethod`` is not supported.

The use of ``classmethod`` is not supported.

Inheritance
~~~~~~~~~~~

Class inhertance is not considered in this proposal.  The only accepted base
class for a jitclass is `object`.

Supported targets
~~~~~~~~~~~~~~~~~~

Only the CPU target (including the parallel target) is supported.
GPUs (e.g. CUDA and HSA) targets are supported via an immutable version of the
jitclass instance, which will be described in a separate NBEP.


Other properties
~~~~~~~~~~~~~~~~

Given:

.. code-block:: python

    spec = [('x', numba.int32),
            ('y', numba.float32)]

    @jitclass(spec)
    class Vec(object):
        ...

* ``isinstance(Vec(1, 2), Vec)`` is True.
* ``type(Vec(1, 2))`` may not be ``Vec``.

Future enhancements
~~~~~~~~~~~~~~~~~~~

This proposal has only described the basic semantic and functionality of a
jitclass.  Additional features will be described in future enhancement
proposals.
