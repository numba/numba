+---------------------------+
| layout: page              |
+---------------------------+
| title: Type (llvm.core)   |
+---------------------------+

llvm.core.Type
==============

-  This will become a table of contents (this text will be scraped).
   {:toc}

Static Constructors
-------------------

``int(n)``
~~~~~~~~~~

Create an integer type of bit width ``n``.

``float()``
~~~~~~~~~~~

Create a 32-bit floating point type.

``double()``
~~~~~~~~~~~~

Create a 64-bit floating point type.

``x86_fp80()``
~~~~~~~~~~~~~~

Create a 80-bit 80x87-style floating point type.

``fp128()``
~~~~~~~~~~~

Create a 128-bit floating point type (112-bit mantissa).

``ppc_fp128()``
~~~~~~~~~~~~~~~

Create a 128-bit float (two 64-bits).

``function(ret, params, vararg=False)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a function type, having the return type ``ret`` (must be a
``Type``), accepting the parameters ``params``, where ``params`` is an
iterable, that yields ``Type`` objects representing the type of each
function argument in order. If ``vararg`` is ``True``, function is
variadic.

``struct(eltys, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an unpacked structure. ``eltys`` is an iterable, that yields
``Type`` objects representing the type of each element in order.

If ``name`` is evaulates ``True`` (not empty), create an *identified
structure*; otherwise, create a *literal structure* by default.

``packed_struct(eltys, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Like ``struct(eltys)``, but creates a packed struct.

``array(elty, count)``
~~~~~~~~~~~~~~~~~~~~~~

Creates an array type, holding ``count`` elements, each of type ``elty``
(which should be a ``Type``).

``pointer(pty, addrspc=0)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a pointer to type ``pty`` (which should be a ``Type``).
``addrspc`` is an integer that represents the address space of the
pointer (see LLVM docs or ask on llvm-dev for more info).

``void()``
~~~~~~~~~~

Creates a void type. Used for function return types.

``label()``
~~~~~~~~~~~

Creates a label type.

``opaque(name)``
~~~~~~~~~~~~~~~~

Opaque `StructType <llvm.core.StructType.html>`_, used for creating
self-referencing types.

Properties
----------

``kind``
~~~~~~~~

[read-only]

A value (enum) representing the "type" of the object. It will be one of
the following constants defined in ``llvm.core``:

{% highlight python %} # Warning: do not rely on actual numerical
values! TYPE\_VOID = 0 TYPE\_FLOAT = 1 TYPE\_DOUBLE = 2 TYPE\_X86\_FP80
= 3 TYPE\_FP128 = 4 TYPE\_PPC\_FP128 = 5 TYPE\_LABEL = 6 TYPE\_INTEGER =
7 TYPE\_FUNCTION = 8 TYPE\_STRUCT = 9 TYPE\_ARRAY = 10 TYPE\_POINTER =
11 TYPE\_OPAQUE = 12 TYPE\_VECTOR = 13 TYPE\_METADATA = 14 TYPE\_UNION =
15 {% endhighlight %}

Example:
^^^^^^^^

{% highlight python %} assert Type.int().kind == TYPE\_INTEGER assert
Type.void().kind == TYPE\_VOID {% endhighlight %}

Methods
-------

``refine``
~~~~~~~~~~

Used for constructing self-referencing types. See the documentation of
`TypeHandle <llvm.core.TypeHandle.html>`_ objects.

Special Methods
---------------

``__str__``
~~~~~~~~~~~

``Type`` objects can be stringified into it's LLVM assembly language
representation.

``__eq__``
~~~~~~~~~~

``Type`` objects can be compared for equality. Internally, this converts
both arguments into their LLVM assembly representations and compares the
resultant strings.
