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

There is currently no high-level API, making some use cases more
complicated than they should be.

Proposed changes
----------------

Dedicated module
''''''''''''''''

We propose the addition of a ``numba.extending`` module exposing the main
APIs useful for extending Numba.

Implementing a function
'''''''''''''''''''''''

We propose the addition of a ``@overload`` decorator allowing the
implementation of a given function for use in :term:`nopython mode`.
The overloading function has the same formal signature as the implemented
function, and receives the actual argument types.  It should return a
Python function implementing the overloaded function for the given types.

The following example implements :func:`numpy.where` with
this approach.

.. literalinclude:: np-where-override.py

It is also possible to implement functions already known to Numba, to
support additional types.  The following example implements the
built-in function :func:`len` for tuples with this approach::

   @overload(len)
   def tuple_len(x):
      if isinstance(x, types.BaseTuple):
         # The tuple length is known at compile-time, so simply reify it
         # as a constant.
         n = len(x)
         def len_impl(x):
            return n
         return len_impl


Implementing an attribute
'''''''''''''''''''''''''

We propose the addition of a ``@overload_attribute`` decorator allowing
the implementation of an attribute getter for use in :term:`nopython mode`.

The following example implements the ``.nbytes`` attribute on Numpy arrays::

   @overload_attribute(types.Array, 'nbytes')
   def array_nbytes(arr):
      def get(arr):
          return arr.size * arr.itemsize
      return get

.. note::
   The overload_attribute() signature allows for expansion to also define
   setters and deleters, by letting the decorated function return a
   ``getter, setter, deleter`` tuple instead of a single ``getter``.


Implementing a method
'''''''''''''''''''''

We propose the addition of a ``@overload_method`` decorator allowing the
implementation of an instance method for use in :term:`nopython mode`.

The following example implements the ``.take()`` method on Numpy arrays::

   @overload_method(types.Array, 'take')
   def array_take(arr, indices):
      if isinstance(indices, types.Array):
          def take_impl(arr, indices):
              n = indices.shape[0]
              res = np.empty(n, arr.dtype)
              for i in range(n):
                  res[i] = arr[indices[i]]
              return res
          return take_impl


Exposing a structure member
'''''''''''''''''''''''''''

We propose the addition of a ``make_attribute_wrapper()`` function exposing
an internal field as a visible read-only attribute, for those types backed
by a ``StructModel`` data model.

For example, assuming ``PdIndexType`` is the Numba type of pandas indices,
here is how to expose the underlying Numpy array as a ``._data`` attribute::

   @register_model(PdIndexType)
   class PdIndexModel(models.StructModel):
       def __init__(self, dmm, fe_type):
           members = [
               ('values', fe_type.as_array),
               ]
           models.StructModel.__init__(self, dmm, fe_type, members)

   make_attribute_wrapper(PdIndexType, 'values', '_data')


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

(when one controls the class being type-inferred, an alternative
to ``typeof_impl`` is to define a ``_numba_type_`` property on the class)

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

Naming of the various decorators is quite vague and confusing.  We propose
renaming ``@builtin`` to ``@infer``, ``@builtin_attr`` to ``@infer_getattr``
and ``builtin_global`` to ``infer_global``.

The two-step declaration for global values is a bit verbose, we propose
simplifying it by allowing the use of ``infer_global`` as a decorator::

   @infer_global(len)
   class Len(AbstractTemplate):
       key = len

       def generic(self, args, kws):
           assert not kws
           (val,) = args
           if isinstance(val, (types.Buffer, types.BaseTuple)):
               return signature(types.intp, val)

The class-based API can feel clumsy, we can add a functional API for
some of the template kinds:

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

Add a generic function for implicit conversion, with multiple dispatch
based on the source and destination types.  Here is an example showing
how to write a float-to-integer conversion::

   @lower_cast(types.Float, types.Integer)
   def float_to_integer(context, builder, fromty, toty, val):
       lty = context.get_value_type(toty)
       if toty.signed:
           return builder.fptosi(val, lty)
       else:
           return builder.fptoui(val, lty)


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

Review and streamine the API.  Drop the requirement to write
``types.Kind(...)`` explicitly.  Remove the separate ``@implement``
decorator and rename ``@builtin`` to ``@lower_builtin``, ``@builtin_attr``
to ``@lower_getattr``, etc.

Add decorators to implement ``setattr()`` operations, named
``@lower_setattr`` and ``@lower_setattr_generic``.


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

Change the implementation signature from ``(c, typ, val)`` to
``(typ, val, c)``, to match the one chosen for the ``typeof_impl``
generic function.
