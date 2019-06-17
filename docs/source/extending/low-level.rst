
.. _low-level-extending:

Low-level extension API
=======================

This extension API is available through the :mod:`numba.extending` module.
It allows you to hook directly into the Numba compilation chain.  As such,
it distinguished between several compilation phases:

* The :term:`typing` phase deduces the types of variables in a compiled
  function by looking at the operations performed.

* The :term:`lowering` phase converts high-level Python operations into
  low-level LLVM code.  This phase exploits the typing information derived
  by the typing phase.

* *Boxing* and *unboxing* convert Python objects into native values, and
  vice-versa.  They occur at the boundaries of calling a Numba function
  from the Python interpreter.


Typing
------

.. XXX the API described here can be insufficient for some use cases.
   Should we describe the whole templates menagerie?

Type inference -- or simply *typing* -- is the process of assigning
Numba types to all values involved in a function, so as to enable
efficient code generation.  Broadly speaking, typing comes in two flavours:
typing plain Python *values* (e.g. function arguments or global variables)
and typing *operations* (or *functions*) on known value types.

.. decorator:: typeof_impl.register(cls)

   Register the decorated function as typing Python values of class *cls*.
   The decorated function will be called with the signature ``(val, c)``
   where *val* is the Python value being typed and *c* is a context
   object.


.. decorator:: type_callable(func)

   Register the decorated function as typing the callable *func*.
   *func* can be either an actual Python callable or a string denoting
   a operation internally known to Numba (for example ``'getitem'``).
   The decorated function is called with a single *context* argument
   and must return a typer function.  The typer function should have
   the same signature as the function being typed, and it is called
   with the Numba *types* of the function arguments; it should return
   either the Numba type of the function's return value, or ``None``
   if inference failed.


Lowering
--------

The following decorators all take a type specification of some kind.
A type specification is usually a type class (such as ``types.Float``)
or a specific type instance (such as ``types.float64``).  Some values
have a special meaning:

* ``types.Any`` matches any type; this allows doing your own dispatching
  inside the implementation

* ``types.VarArg(<some type>)`` matches any number of arguments of the
  given type; it can only appear as the last type specification when
  describing a function's arguments.

A *context* argument in the following APIs is a target context providing
various utility methods for code generation (such as creating a constant,
converting from a type to another, looking up the implementation of a
specific function, etc.).  A *builder* argument is a
:class:`llvmlite.ir.IRBuilder` instance for the LLVM code being generated.

A *signature* is an object specifying the concrete type of an operation.
The ``args`` attribute of the signature is a tuple of the argument types.
The ``return_type`` attribute of the signature is the type that the
operation should return.

.. note::
   Numba always reasons on Numba types, but the values being passed
   around during lowering are LLVM values: they don't hold the required
   type information, which is why Numba types are passed explicitly too.

   LLVM has its own, very low-level type system: you can access the LLVM
   type of a value by looking up its ``.type`` attribute.


Native operations
'''''''''''''''''

.. decorator:: lower_builtin(func, typespec, ...)

   Register the decorated function as implementing the callable *func*
   for the arguments described by the given Numba *typespecs*.
   As with :func:`type_callable`, *func* can be either an actual Python
   callable or a string denoting a operation internally known to Numba
   (for example ``'getitem'``).

   The decorated function is called with four arguments
   ``(context, builder, sig, args)``.  ``sig`` is the concrete signature
   the callable is being invoked with.  ``args`` is a tuple of the values
   of the arguments the callable is being invoked with; each value in
   ``args`` corresponds to a type in ``sig.args``.  The function
   must return a value compatible with the type ``sig.return_type``.

.. decorator:: lower_getattr(typespec, name)

   Register the decorated function as implementing the attribute *name*
   of the given *typespec*.  The decorated function is called with four
   arguments ``(context, builder, typ, value)``.  *typ* is the concrete
   type the attribute is being looked up on.  *value* is the value the
   attribute is being looked up on.

.. decorator:: lower_getattr_generic(typespec)

   Register the decorated function as a fallback for attribute lookup
   on a given *typespec*.  Any attribute that does not have a corresponding
   :func:`lower_getattr` declaration will go through
   :func:`lower_getattr_generic`.  The decorated function is called with
   five arguments ``(context, builder, typ, value, name)``.  *typ*
   and *value* are as in :func:`lower_getattr`.  *name* is the name
   of the attribute being looked up.

.. decorator:: lower_cast(fromspec, tospec)

   Register the decorated function as converting from types described by
   *fromspec* to types described by *tospec*.  The decorated function
   is called with five arguments ``(context, builder, fromty, toty, value)``.
   *fromty* and *toty* are the concrete types being converted from and to,
   respectively.  *value* is the value being converted.  The function
   must return a value compatible with the type ``toty``.


Constants
'''''''''

.. decorator:: lower_constant(typespec)

   Register the decorated function as implementing the creation of
   constants for the Numba *typespec*.  The decorated function
   is called with four arguments ``(context, builder, ty, pyval)``.
   *ty* is the concrete type to create a constant for.  *pyval*
   is the Python value to convert into a LLVM constant.
   The function must return a value compatible with the type ``ty``.


Boxing and unboxing
'''''''''''''''''''

In these functions, *c* is a convenience object with several attributes:

* its ``context`` attribute is a target context as above
* its ``builder`` attribute is a :class:`llvmlite.ir.IRBuilder` as above
* its ``pyapi`` attribute is an object giving access to a subset of the
  `Python interpreter's C API <https://docs.python.org/3/c-api/index.html>`_

An object, as opposed to a native value, is a ``PyObject *`` pointer.
Such pointers can be produced or processed by the methods in the ``pyapi``
object.

.. decorator:: box(typespec)

   Register the decorated function as boxing values matching the *typespec*.
   The decorated function is called with three arguments ``(typ, val, c)``.
   *typ* is the concrete type being boxed.  *val* is the value being
   boxed.  The function should return a Python object, or NULL to signal
   an error.

.. decorator:: unbox(typespec)

   Register the decorated function as unboxing values matching the *typespec*.
   The decorated function is called with three arguments ``(typ, obj, c)``.
   *typ* is the concrete type being unboxed.  *obj* is the Python object
   (a ``PyObject *`` pointer, in C terms) being unboxed.  The function
   should return a ``NativeValue`` object giving the unboxing result value
   and an optional error bit.
