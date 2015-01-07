
============================
Extending the Numba Frontend
============================

.. warning::
   The Numba APIs described in this document are not guaranteed to be stable.
   External packages that rely on these APIs may break with new Numba releases.
   Their description is mostly useful in the context of extending Numba
   withing the Numba codebase.

Overview
========

The frontend of Numba first analyzes the control and data flow of a function.
It then performs type inference to deduce the types of all intermediate values
and identify points where types must be coerced. Type inference attempts to
determine a single specific type for variable. When a variable's type cannot be
deduced, or it is determined to take on multiple specific types depending on the
control flow, its type falls back to ``pyobject``.

The frontend must succeed in typing all variables unambiguously (i.e. they must
not be typed as ``pyobject``) in order for the backend to generate code in
nopython mode, because the backend uses type information to match appropriate
code generators with the values they operate on. Extending the frontend
primarily consists of adding support for new types, to allow variables that hold
instances of these types to be typed unambiguously.

Numba Types
-----------

All Numba types are instances of classes that inherit from ``numba.types.Type``.
Numba types can be parameterized (for example, arrays and records), in which
case their Type classes will take constructor arguments defining the parameters.
Different instances of a parameterized type usually denote distinct types and
can trigger different, specialized code generation in the backend.

.. note::
   In the rest of this document, when we refer to a "type", we mean the
   Numba type unless we explicitly write "Python type".

Type Inference Mechanism
------------------------

Type inference is performed for variables in three cases:

* When automatic JIT compilation is used: the types of function arguments must
  be deduced from the values passed in.
* When global variables are accessed: the Numba types of those globals is
  deduced from their values at the time of compilation.
* Intermediate values: within a function, the type of every intermediate
  variable must be deduced.

The types of intermediate values are determined by iteratively propagating type
information through the data flow graph (DFG). Each iteration propagates type
information along the edges of the DFG, until convergence is reached. When two
edges flow into the same node and differing type information is propagated, the
type of the node is resolved as ``pyobject``.

In order for the propagation to proceed through functions, operators, and
attributes, Numba needs to make use of *Type Signatures*, which map input types
to output types. Numba needs type signatures for:

* Object attributes: This can include the attributes of instances of Python
  classes, or modules.
* Global values: Objects (such as functions) accessed from the global
  namespace.
* Operators and other "implicit" functions: Certain Python syntax
  (like ``a + b``, or  ``iter(o)``) triggers special function calls.
  To overload these operations, a type signature for the appropriate function
  must be registered.
* Other entities not described in this document, such as builtin functions.

All type inference happens with a *Typing Context*. Each target has its own
Typing Context - presently there are two, for the CPU and CUDA backends. The
majority of type signatures are common between these contexts, but the creation
of a context for each target allows specialisation based on intrinsics or other
specialised operations and types that a target may support.

Tutorial
========

We will extend the Numba frontend to support typing a class that it does not
currently support by:

* Adding a Numba Type corresponding to the class,
* Adding the relevant type signatures for a function and an attribute of the
  class, and
* Adding a type signature for overloading an elementary operation.

The example will add support for a module named ``interval``, which is assumed
to be external to Numba and contains the following::

    class Interval(object):
        '''A half-open interval on the real number line.'''
        def __init__(self, lo, hi):
            self.lo = lo
            self.hi = hi

        def __repr__(self):
            return 'Interval(%f, %f)' % (self.lo, self.hi)

    def valid_interval(interval):
        '''Return True if interval.lo <= interval.hi'''
        return interval.lo <= interval.hi

Creating a New Numba Type
-------------------------

Types are defined in the ``numba.types`` module.  To create a new Numba type,
subclass ``numba.types.Type`` and make a single instance of it::

    class IntervalType(numba.types.Type):
        def __init__(self):
            super(IntervalType, self).__init__(name='Interval')

    interval_type = IntervalType()

This enables ``interval_type`` to be used to declare argument and return types
in ``@jit`` decorations. For example::

    @jit(numba.types.bool_(interval_type, numba.types.float32))
    def inside(interval, x):
        return interval.lo <= x < interval.hi

Organizing Type Signatures with a Registry
------------------------------------------

Numba uses a *Registry* (class ``numba.typing.templates.Registry``) to hold
collections of related type signatures for attributes, globals and operators.

Examples of the use of a Registry can be found
in ``numba.typing.cmathdecl``, ``numba.typing.npydecl``, and some other modules
in ``numba.typing``.

For our ``interval`` example, we will create a new Registry. This is overkill
for a small set of type signatures, but is representative of what would be
required when adding type signatures for more complicated classes and modules.

We will create the ``numba.typing.intervaldecl`` module and add the following::

    from numba.typing.templates import Registry

    registry = Registry()
    register = registry.register
    register_attr = registry.register_attr
    register_global = registry.register_global

``register``, ``register_attr``, and ``register_global`` may now be used later
in the module as decorators to record functions that compute the type signatures
of functions, attributes, and globals, respectively.

Adding an Attribute Value Type Signature
----------------------------------------

We can add type signatures for attributes of instances of ``Interval``, so
that ``lo`` and ``hi`` are recognized as returning ``float32`` types.  This
requires creating a subclass of ``numba.typing.templates.AttributeTemplate``
(add the following to ``numba.typing.intervaldecl``)::

    from numba.types import float32
    from numba.typing.templates import AttributeTemplate

    @register_attr
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
``valid_interval`` global function, we need to provide a function type signature
for it.  This is done using a ``numba.typing.templates.ConcreteTemplate``. Add
the following to ``numba.typing.intervaldecl``::

    from numba.types import bool_, Function
    from numba.typing.templates import ConcreteTemplate, signature
    from interval import valid_interval

    class ValidIntervalSignature(ConcreteTemplate):
        key = valid_interval
        cases = [
            signature(bool_, interval_type)
        ]

    register_global(valid_interval, Function(ValidIntervalSignature))

The ``key`` for looking up the function type is the Python function itself,
``valid_interval`` in this example.  The ``cases`` attribute lists all of the
supported function signature combinations.  The first argument to
``signature`` is the return type, and the remaining arguments are the types of
the function arguments.  Only positional arguments are supported for function
types (i.e. no keyword arguments).

Overloading Elementary Operations
---------------------------------

Next, suppose we want to add support for a ``+`` operation between two
intervals.  We need to make a ``ConcreteTemplate`` where the key is the string
``"+"``. Add to ``numba.typing.intervaldecl``::

    from numba.typing.templates import ConcreteTemplate

    @builtin
    class AdditionSignature(ConcreteTemplate):
        key = '+'
        cases = [
            signature(interval_type, interval_type, interval_type)
        ]

Several templates with the same key can be inserted, and each will be checked
for a matching function signatures in the order of insertion. This allows the
same key to be overloaded with different numbers of arguments and different
argument types.

The list of special function keys includes:

.. todo:: correct this list

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

Installing the Registry in a Typing Context
-------------------------------------------

Once all required type signatures have been added to a Registry, it can then be
installed into a typing context. In this example, we will make the registry that
we have created available to all typing contexts, so we will make sure that it
is installed by modifying ``numba.typing.context.BaseContext``::

    class Context(BaseContext):
        def init(self):
            self.install(cmathdecl.registry)
            self.install(intervaldecl.registry)
            self.install(mathdecl.registry)
            self.install(npydecl.registry)
            self.install(operatordecl.registry)

Note the addition of the installation of ``intervaldecl.registry``.

Enabling Type Inference for Function Arguments and Globals
----------------------------------------------------------

Numba is infers the types of arguments and global variables, using the
``BaseContext.resolve_data_type`` method. In order to add support for the
``Interval`` class, we must first create a function that detects ``Interval``
instances. Create a new module, ``numba.interval_support``, containing::

    import interval

    def is_interval(typ):
        return isinstance(typ, interval.Interval)

Then modify the ``BaseContext.get_data_type`` function in
``numba.typing.context`` so that just before the final ``return`` statement, the
following check is added::

    if interval_support.is_interval(val):
        return types.interval_type

and add an import for ``numba.interval_support`` to the top of the file.

Conclusion
==========

So far we have added support for typing for an attribute, a function, an
elementary operator, and have added type inference for function arguments and
globals. However, this does not yet enable any change in the code generated by
Numba, which requires the addition of backend support for the ``Interval``
class, described in the next section.
