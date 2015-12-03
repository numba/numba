========================
NBEP 2: Extension points
========================

:Author: Antoine Pitrou
:Date: July 2015
:Status: Draft


Implementing new types or functions in Numba requires hooking into
various mechanisms along the compilation chain (and potentially
outside of it).  This document aims, first, at examining the
current ways of doing so and, second, at making proposals to make
extending easier.

If some of the proposals are implemented, we should first strive
to use and exercise them internally, before exposing the APIs to the
public.

.. note::
   This document doesn't cover CUDA or any other non-CPU backend.


High-level API
==============

There is currently no high-level API for quick implementation of
an existing function or type.

Proposed changes
----------------

It would be nice for people to be able to implement a function
in a single go, as if they were writing a ``@jit`` function.
As an example, let's assume we want to make :func:`numpy.where`
usable from :term:`nopython mode`.  We would like to be able
to define several implementations and select between them at
compile-time depending on the input types.

The following example showcases a hypothetical API where we can
register a function taking the argument types and returning a
callable implementing the actual function for those types.
The API should also be able to handle optional arguments, and
the resulting implementation should support calling with named
parameters.

.. literalinclude:: np-where-override.py


Typing
======

Numba types
-----------

Numba's standard types are declared in :mod:`numba.types`.  To declare
a new type, one subclasses the base :class:`Type` class or one of its
existing abstract subclasses, and implements the required functionality.

Proposed changes
''''''''''''''''

No change required.


Type inference on values
------------------------

Values of a new type need to be type-inferred if they can appear as
function arguments or constants.  The core machinery is in
:mod:`numba.typing.typeof`.

In the common case where some Python class or classes map exclusively
to the new type, one can extend a generic function to dispatch on said
classes, e.g.::

   from numba.typing.typeof import typeof_impl

   @typeof_impl(MyClass)
   def _typeof_myclass(val, c):
      if "some condition":
         return MyType(...)

The ``typeof_impl`` specialization must return a Numba type instance,
or None if the value failed typing.

In the rarer case where the new type can denote various Python classes
that are impossible to enumerate, one must insert a manual check in the
fallback implementation of the ``typeof_impl`` generic function.

Proposed changes
''''''''''''''''

Allow people to define a generic hook without monkeypatching the
fallback implementation.


Fast path for type inference on function arguments
--------------------------------------------------

Optionally, one may want to allow a new type to participate in the
fast type resolution (written in C code) to minimize function call
overhead when a JIT-compiled function is called with the new type.
One must then insert the required checks and implementation in
the ``_typeof.c`` file, presumably inside the ``compute_fingerprint()``
function.

Proposed changes
''''''''''''''''

None.  Adding generic hooks to C code embedded in a C Python extension
is too delicate a change.


Type inference on operations
----------------------------

Values resulting from various operations (function calls, operators, etc.)
are typed using a set of helpers called "templates".  One can define a
new template by subclass one of the existing base classes and implement
the desired inference mechanism.  The template is explicitly registered
with the type inference machinery using a decorator.

The :class:`ConcreteTemplate` base class allows one to define inference as
a set of supported signatures for a given operation.  The following example
types the modulo operator::

   @builtin
   class BinOpMod(ConcreteTemplate):
       key = "%"
       cases = [signature(op, op, op)
                for op in sorted(types.signed_domain)]
       cases += [signature(op, op, op)
                 for op in sorted(types.unsigned_domain)]
       cases += [signature(op, op, op) for op in sorted(types.real_domain)]

(note that type *instances* are used in the signatures, severely
limiting the amount of genericity that can be expressed)

The :class:`AbstractTemplate` base class allows to define inference
programmatically, giving it full flexibility.  Here is a simplistic
example of how tuple indexing (i.e. the ``__getitem__`` operator) can
be expressed::

   @builtin
   class GetItemUniTuple(AbstractTemplate):
       key = "getitem"

       def generic(self, args, kws):
           tup, idx = args
           if isinstance(tup, types.UniTuple) and isinstance(idx, types.Integer):
               return signature(tup.dtype, tup, idx)


The :class:`AttributeTemplate` base class allows to type the attributes
and methods of a given type.  Here is an example, typing the ``.real``
and ``.imag`` attributes of complex numbers::

   @builtin_attr
   class ComplexAttribute(AttributeTemplate):
       key = types.Complex

       def resolve_real(self, ty):
           return ty.underlying_float

       def resolve_imag(self, ty):
           return ty.underlying_float

.. note::
   :class:`AttributeTemplate` only works for getting attributes.  Setting
   an attribute's value is hardcoded in :mod:`numba.typeinfer`.

The :class:`CallableTemplate` base class offers an easier way to parse
flexible function signatures, by letting one define a callable that has
the same definition as the function being typed.  For example, here is how
one could hypothetically type Python's ``sorted`` function if Numba supported
lists::

   @builtin
   class Sorted(CallableTemplate):
       key = sorted

       def generic(self):
           def typer(iterable, key=None, reverse=None):
               if reverse is not None and not isinstance(reverse, types.Boolean):
                   return
               if key is not None and not isinstance(key, types.Callable):
                   return
               if not isinstance(iterable, types.Iterable):
                   return
               return types.List(iterable.iterator_type.yield_type)

           return typer

(note you can return just the function's return type instead of the
full signature)

Proposed changes
''''''''''''''''

If we expose some of this, should we streamline the API first?
The class-based API can feel clumsy, one could instead imagine
a functional API for some of the template kinds:

.. code-block:: python

   @type_callable(sorted)
   def type_sorted(context):
       def typer(iterable, key=None, reverse=None):
           # [same function as above]

       return typer


Code generation
===============

Concrete representation of values of a Numba type
-------------------------------------------------

Any concrete Numba type must be able to be represented in LLVM form
(for variable storage, argument passing, etc.).  One defines that
representation by implementing a datamodel class and registering it
with a decorator.  Datamodel classes for standard types are defined
in :mod:`numba.datamodel.models`.

Proposed changes
''''''''''''''''

No change required.

Conversion between types
------------------------

Implicit conversion between Numba types is currently implemented as a
monolithic sequence of choices and type checks in the
:meth:`BaseContext.cast` method.  To add a new implicit conversion, one
appends a type-specific check in that method.

Boolean evaluation is a special case of implicit conversion (the
destination type being :class:`types.Boolean`).

.. note::
   Explicit conversion is seen as a regular operation, e.g. a constructor
   call.

Proposed changes
''''''''''''''''

Implicit conversion could use some kind of generic function, with multiple
dispatch based on the source and destination types.

Implementation of an operation
------------------------------

Other operations are implemented and registered using a set of generic
functions and decorators.  For example, here is how lookup for a the ``.ndim``
attribute on Numpy arrays is implemented::

   @builtin_attr
   @impl_attribute(types.Kind(types.Array), "ndim", types.intp)
   def array_ndim(context, builder, typ, value):
       return context.get_constant(types.intp, typ.ndim)

And here is how calling ``len()`` on a tuple value is implemented::

   @builtin
   @implement(types.len_type, types.Kind(types.BaseTuple))
   def tuple_len(context, builder, sig, args):
       tupty, = sig.args
       retty = sig.return_type
       return context.get_constant(retty, len(tupty.types))

Proposed changes
''''''''''''''''

No changes required.  Perhaps review and streamine the API (drop the
requirement to write ``types.Kind(...)`` explicitly?).


Conversion from / to Python objects
-----------------------------------

Some types need to be converted from or to Python objects, if they can
be passed as function arguments or returned from a function.  The
corresponding boxing and unboxing operations are implemented using
a generic function.  The implementations for standard Numba types
are in :mod:`numba.targets.boxing`.  For example, here is the boxing
implementation for a boolean value::

   @box(types.Boolean)
   def box_bool(c, typ, val):
       longval = c.builder.zext(val, c.pyapi.long)
       return c.pyapi.bool_from_long(longval)

Proposed changes
''''''''''''''''

Perhaps change the implementation signature slightly, from ``(c, typ, val)``
to ``(typ, val, c)``, to match the one chosen for the ``typeof_impl``
generic function.
