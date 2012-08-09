+-------------------------------+
| layout: page                  |
+-------------------------------+
| title: Constant (llvm.core)   |
+-------------------------------+

``Constant``-s represents constants that appear within the code. The
values of such objects are known at creation time. Constants can be
created from Python constants. A constant expression is also a constant
-- given a ``Constant`` object, an operation (like addition, subtraction
etc) can be specified, to yield a new ``Constant`` object. Let's see
some examples:

{% highlight python %} #!/usr/bin/env python

ti = Type.int() # a 32-bit int type

k1 = Constant.int(ti, 42) # "int k1 = 42;" k2 = k1.add( Constant.int(
ti, 10 ) ) # "int k2 = k1 + 10;"

tr = Type.float()

r1 = Constant.real(tr, "3.141592") # create from a string r2 =
Constant.real(tr, 1.61803399) # create from a Python float {%
endhighlight %}

llvm.core.Constant
==================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Static factory methods
----------------------

``null(ty)``
~~~~~~~~~~~~

A null value (all zeros) of type ``ty``

``all_ones(ty)``
~~~~~~~~~~~~~~~~

All 1's value of type ``ty``

``undef(ty)``
~~~~~~~~~~~~~

An undefined value of type ``ty``

``int(ty, value)``
~~~~~~~~~~~~~~~~~~

Integer of type ``ty``, with value ``value`` (a Python int or long)

``int_signextend(ty, value)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Integer of signed type ``ty`` (use for signed types)

``real(ty, value)``
~~~~~~~~~~~~~~~~~~~

Floating point value of type ``ty``, with value ``value`` (a Python
float)

``stringz(value)``
~~~~~~~~~~~~~~~~~~

A null-terminated string. ``value`` is a Python string

``string(value)``
~~~~~~~~~~~~~~~~~

As ``string(ty)``, but not null terminated

``array(ty, consts)``
~~~~~~~~~~~~~~~~~~~~~

Array of type ``ty``, initialized with ``consts`` (an iterable yielding
``Constant`` objects of the appropriate type)

``struct(ty, consts)``
~~~~~~~~~~~~~~~~~~~~~~

Struct (unpacked) of type ``ty``, initialized with ``consts`` (an
iterable yielding ``Constant`` objects of the appropriate type)

``packed_struct(ty, consts)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As ``struct(ty, consts)`` but packed

``vector(consts)``
~~~~~~~~~~~~~~~~~~

Vector, initialized with ``consts`` (an iterable yielding ``Constant``
objects of the appropriate type)

``sizeof(ty)``
~~~~~~~~~~~~~~

Constant value representing the sizeof the type ``ty``

Methods
-------

The following operations on constants are supported. For more details on
any operation, consult the `Constant
Expressions <http://www.llvm.org/docs/LangRef.html#constantexprs>`_
section of the LLVM Language Reference.

``k.neg()``
~~~~~~~~~~~

negation, same as ``0 - k``

``k.not_()``
~~~~~~~~~~~~

1's complement of ``k``. Note trailing underscore.

``k.add(k2)``
~~~~~~~~~~~~~

``k + k2``, where ``k`` and ``k2`` are integers.

``k.fadd(k2)``
~~~~~~~~~~~~~~

``k + k2``, where ``k`` and ``k2`` are floating-point.

``k.sub(k2)``
~~~~~~~~~~~~~

``k - k2``, where ``k`` and ``k2`` are integers.

``k.fsub(k2)``
~~~~~~~~~~~~~~

``k - k2``, where ``k`` and ``k2`` are floating-point.

``k.mul(k2)``
~~~~~~~~~~~~~

``k * k2``, where ``k`` and ``k2`` are integers.

``k.fmul(k2)``
~~~~~~~~~~~~~~

``k * k2``, where ``k`` and ``k2`` are floating-point.

``k.udiv(k2)``
~~~~~~~~~~~~~~

Quotient of unsigned division of ``k`` with ``k2``

``k.sdiv(k2)``
~~~~~~~~~~~~~~

Quotient of signed division of ``k`` with ``k2``

``k.fdiv(k2)``
~~~~~~~~~~~~~~

Quotient of floating point division of ``k`` with ``k2``

``k.urem(k2)``
~~~~~~~~~~~~~~

Reminder of unsigned division of ``k`` with ``k2``

``k.srem(k2)``
~~~~~~~~~~~~~~

Reminder of signed division of ``k`` with ``k2``

``k.frem(k2)``
~~~~~~~~~~~~~~

Reminder of floating point division of ``k`` with ``k2``

``k.and_(k2)``
~~~~~~~~~~~~~~

Bitwise and of ``k`` and ``k2``. Note trailing underscore.

``k.or_(k2)``
~~~~~~~~~~~~~

Bitwise or of ``k`` and ``k2``. Note trailing underscore.

``k.xor(k2)``
~~~~~~~~~~~~~

Bitwise exclusive-or of ``k`` and ``k2``.

``k.icmp(icmp, k2)``
~~~~~~~~~~~~~~~~~~~~

Compare ``k`` with ``k2`` using the predicate ``icmp``. See
`here <comparision.html#icmp>`_ for list of predicates for integer
operands.

``k.fcmp(fcmp, k2)``
~~~~~~~~~~~~~~~~~~~~

Compare ``k`` with ``k2`` using the predicate ``fcmp``. See
`here <comparision.html#fcmp>`_ for list of predicates for real
operands.

``k.shl(k2)``
~~~~~~~~~~~~~

Shift ``k`` left by ``k2`` bits.

``k.lshr(k2)``
~~~~~~~~~~~~~~

Shift ``k`` logically right by ``k2`` bits (new bits are 0s).

``k.ashr(k2)``
~~~~~~~~~~~~~~

Shift ``k`` arithmetically right by ``k2`` bits (new bits are same as
previous sign bit).

``k.gep(indices)``
~~~~~~~~~~~~~~~~~~

GEP, see `LLVM docs <http://www.llvm.org/docs/GetElementPtr.html>`_.

``k.trunc(ty)``
~~~~~~~~~~~~~~~

Truncate ``k`` to a type ``ty`` of lower bitwidth.

``k.sext(ty)``
~~~~~~~~~~~~~~

Sign extend ``k`` to a type ``ty`` of higher bitwidth, while extending
the sign bit.

``k.zext(ty)``
~~~~~~~~~~~~~~

Sign extend ``k`` to a type ``ty`` of higher bitwidth, all new bits are
0s.

``k.fptrunc(ty)``
~~~~~~~~~~~~~~~~~

Truncate floating point constant ``k`` to floating point type ``ty`` of
lower size than k's.

``k.fpext(ty)``
~~~~~~~~~~~~~~~

Extend floating point constant ``k`` to floating point type ``ty`` of
higher size than k's.

``k.uitofp(ty)``
~~~~~~~~~~~~~~~~

Convert an unsigned integer constant ``k`` to floating point constant of
type ``ty``.

``k.sitofp(ty)``
~~~~~~~~~~~~~~~~

Convert a signed integer constant ``k`` to floating point constant of
type ``ty``.

``k.fptoui(ty)``
~~~~~~~~~~~~~~~~

Convert a floating point constant ``k`` to an unsigned integer constant
of type ``ty``.

``k.fptosi(ty)``
~~~~~~~~~~~~~~~~

Convert a floating point constant ``k`` to a signed integer constant of
type ``ty``.

``k.ptrtoint(ty)``
~~~~~~~~~~~~~~~~~~

Convert a pointer constant ``k`` to an integer constant of type ``ty``.

``k.inttoptr(ty)``
~~~~~~~~~~~~~~~~~~

Convert an integer constant ``k`` to a pointer constant of type ``ty``.

``k.bitcast(ty)``
~~~~~~~~~~~~~~~~~

Convert ``k`` to a (equal-width) constant of type ``ty``.

``k.select(cond,k2,k3)``
~~~~~~~~~~~~~~~~~~~~~~~~

Replace value with ``k2`` if the 1-bit integer constant ``cond`` is 1,
else with ``k3``.

``k.extract_element(idx)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract value at ``idx`` (integer constant) from a vector constant
``k``.

``k.insert_element(k2,idx)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert value ``k2`` (scalar constant) at index ``idx`` (integer
constant) of vector constant ``k``.

``k.shuffle_vector(k2,mask)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Shuffle vector constant ``k`` based on vector constants ``k2`` and
``mask``.

--------------

Other Constant Classes
======================

The following subclasses of ``Constant`` do not provide additional
methods, **they serve only to provide richer type information.**

Subclass \| LLVM C++ Class \| Remarks \|
---------\|----------------\|---------\| ``ConstantExpr`` \|
``llvmConstantExpr`` \| A constant expression \|
``ConstantAggregateZero``\ \| ``llvmConstantAggregateZero``\ \| All-zero
constant \| ``ConstantInt``\ \| ``llvmConstantInt``\ \| An integer
constant \| ``ConstantFP``\ \| ``llvmConstantFP``\ \| A floating-point
constant \| ``ConstantArray``\ \| ``llvmConstantArray``\ \| An array
constant \| ``ConstantStruct``\ \| ``llvmConstantStruct``\ \| A
structure constant \| ``ConstantVector``\ \| ``llvmConstantVector``\ \|
A vector constant \| ``ConstantPointerNull``\ \|
``llvmConstantPointerNull``\ \| All-zero pointer constant \|
``UndefValue``\ \| ``llvmUndefValue``\ \| corresponds to ``undef`` of
LLVM IR \|

These types are helpful in ``isinstance`` checks, like so:

{% highlight python %} ti = Type.int(32) k1 = Constant.int(ti, 42) #
int32\_t k1 = 42; k2 = Constant.array(ti, [k1, k1]) # int32\_t k2[] = {
k1, k1 };

assert isinstance(k1, ConstantInt) assert isinstance(k2, ConstantArray)
{% endhighlight %}
