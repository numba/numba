============================
Extending the Numba Frontend
============================

.. warning:: The Numba APIs described in this document are not currently guaranteed to be stable.  External packages that rely on these APIs may break with new Numba releases.

Concepts
========

The frontend of Numba analyzes the control flow of a function and performs 
type inference in order to (attempt to) deduce the types of all intermediate
values in the function and identify points where types must be coerced.

Numba Types
-----------

A Numba type is really a category label for values that is used by the back-
end to match appropriate code generators with the values they operate on. All
Numba types are instances of classes that inherit from ``numba.types.Type``.
Numba types can be parameterized (for example, arrays and records), in which
case their Type classes will take constructor arguments defining the
parameters.  Different instances of a parameterized type usually denote
distinct types and can trigger different, specialized code generation in the
backend.

.. note:: In the rest of this document, when we refer to a "type", we mean the Numba type unless we explicitly write "Python type".

Mapping Python Types to Numba Types
-----------------------------------

Although the ``@jit`` decorator allows explicit declarations of the Numba
types in a function signature, sometimes Numba needs to infer the Numba type
associated with a particular Python type.  If automatic JIT compilation is
being used, then Numba will determine the types of function arguments from
the Python values passed as function arguments.  Additionally, if a
function accesses global variables, Numba types will also be inferred from
the Python values of those globals.

Type Signatures
---------------

Once the types of all the externally defined values (function arguments and
globals) have been specified, the type inference engine needs to propagate
these types through all of the expressions in the function.

Numba needs type signatures for:

* Object attributes: This can include the attributes of instances of Python 
  classes, or modules.
* Global values: Objects (such as functions) accessed from the global 
  namespace.
* Operators and other "implicit" functions: Certain Python syntax 
  (like ``a + b``, or  ``iter(o)``) triggers special function calls.
  To overload these operations, a type signature for the appropriate function
  must be registered.
* Other entities not described in this document, such as builtin functions.


Tasks
=====

All of the tasks below will work with an example class and function::

    class Interval(object):
        '''A half-open interval on the real number line.'''
        def __init__(self, lo, hi):
            self.lo = lo
            self.hi = hi

        def __repr__(self):
            return 'Interval(%f, %f)' % (self.lo, self.hi)

    # global function
    def valid_interval(interval):
        '''Return True if interval.lo <= interval.hi'''
        pass  # This is a stub.  We will implement the function in LLVM


Organizing Type Signatures with a Registry
------------------------------------------

If you have a lot of type signatures in a module, it can be cumbersome to make
type information easily portable between targets. The
``numba.typing.templates.Registry`` class simplifies this process by
collecting lists of attribute, global and operator type signatures that can be
installed into a typing context all at once.

A common pattern in the Numba code is to collect all the type information for
a particular package into a module that begins with::

    from numba.typing.templates import (AttributeTemplate, ConcreteTemplate,
                                        signature, Registry)

    registry = Registry()  # A new registry for our new set of types
    builtin = registry.register
    builtin_attr = registry.register_attr
    builtin_global = registry.register_global

Then those three functions are used to record different type signatures in
the registry (see examples below).  When the registry is fully populated,
it is installed in the typing context::

    from numba.targets.registry import target_registry

    # Assuming the CPU target
    target = target_registry['cpu']
    target.targetdescr.typing_context.install(registry)


Creating a New Numba Type
-------------------------

To create a new Numba type, subclass ``numba.types.Type`` and make a single
instance of it::

    class IntervalType(numba.types.Type):
        def __init__(self):
            super(IntervalType, self).__init__(name='Interval')
    interval_type = IntervalType()

``interval_type`` can now be used to declare argument and return types in 
``@jit`` decorations::

    @jit(numba.types.bool_(interval_type, numba.types.float32))
    def inside(interval, x):
        return interval.lo <= x < interval.hi

.. note:: The string form of the JIT signature ``@jit("bool_(interval_type, float32)")`` cannot be used in the above example unless ``interval_type`` has been added to the ``numba.types`` module.  This shortcoming will be fixed in a future Numba version.


Adding an Attribute Value Type Signature
----------------------------------------

We can add type signatures for attributes of instances of ``Interval``, so
that ``lo`` and ``hi`` are recognized as returning ``float32`` types.  This
requires creating a subclass of ``numba.typing.templates.AttributeTemplate``::

    from numba.types import float32
    from numba.typing.templates import AttributeTemplate

    @builtin_attr
    class IntervalAttributes(AttributeTemplate):
        key = interval_type

        # We will store the interval bounds as 32-bit floats
        _attributes = dict(lo=float32, hi=float32)

        def generic_resolve(self, value, attr):
            return self._attributes[attr]

The ``key`` attribute of the template contains the Numba type that needs to be
matched to use this template.  It can either be an instance of a ``Type``
subclass, or the subclass itself, for parametric types.

The ``AttributeTemplate`` will first look for a method of the form
``resolve_<attribute name>`` to get the type of a specific attribute,
otherwise it will delegate to the ``generic_resolve()`` method.  This call
takes both the Numba type instance (useful for parametric types) of the value
being accessed, and the name of the attribute.  The return value from
``generic_resolve()`` is the type of the value returned by the attribute
access.


Adding a Function Type Signature
--------------------------------

In order for the Numba type inference engine to recognize the
``valid_interval`` global function, we need to provide a type signature for
it.  This is done using a ``numba.typing.templates.ConcreteTemplate``::

    from numba.types import bool_, Function
    from numba.targets.registry import target_registry
    from numba.typing.templates import ConcreteTemplate, signature

    # Assuming the CPU target
    target = target_registry['cpu']
    typing_context = target.targetdescr.typing_context

    class ValidIntervalSignature(ConcreteTemplate):
        key = valid_interval
        cases = [
            signature(bool_, interval_type)
        ]

    builtin_global(valid_interval, Function(ValidIntervalSignature))

The ``key`` for looking up the function type is the Python function itself,
``valid_interval`` in this example.  The ``cases`` attribute lists all of the
supported function signature combinations.  The first argument to
``signature`` is the return type, and the remaining arguments are the types of
the function arguments.  Only positional arguments are supported for function
types (i.e. no keyword arguments).


Overloading Elementary Operations
---------------------------------

Suppose we want to add support for a ``+`` operation between two intervals.
We need to make a ``ConcreteTemplate`` where the key is the string ``"+"``::

    from numba.targets.registry import target_registry
    from numba.typing.templates import ConcreteTemplate, signature

    # Assuming the CPU target
    target = target_registry['cpu']
    typing_context = target.targetdescr.typing_context

    @builtin
    class AdditionSignature(ConcreteTemplate):
        key = '+'
        cases = [
            signature(interval_type, interval_type, interval_type)
        ]

Several templates with the same key can be inserted, and each will be checked
for a matching function signatures in the order of insertion. This is what
allows the same key to be overloaded with different numbers of arguments and
different argument types.

The list of special function keys includes:

============    ============
Key             Description
============    ============
``+``           Addition (2 args) and unary positive (1 arg)
``-``           Subtraction (2 args) and unary negative (1 arg)
``*``           Multiplication
``/?``          Divide (only Python 2)
``/``           True divide
``//``          Floor divide
``%``           Modulo
``**``          Power
``<<``          Left shift
``>>``          Right shift
``&``           Bitwise AND
``|``           Bitwise OR
``^``           Bitwise XOR
``getiter``     Get an iterator (equivalent to ``__iter__()``)
``iternext``    Return the next element from an iterator (equivalent to ``__next__()``)
``getitem``     Get an item (equivalent to ``__getitem__()``)
============    ============

These keys come directly from operations in the Numba IR (see :ref:`arch_generate_numba_ir`).

In-place operations (like ``a += b``) are assumed to have the same signature
as the right-hand side of the expanded form (``a = a + b``).


