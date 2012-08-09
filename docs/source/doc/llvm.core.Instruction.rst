+----------------------------------+
| layout: page                     |
+----------------------------------+
| title: Instruction (llvm.core)   |
+----------------------------------+

An ``llvm.core.Instruction`` object represents an LLVM instruction. This
class is the root of a small hierarchy:

::

    Instruction
        CallOrInvokeInstruction
        PHINode
        SwitchInstruction
        CompareInstruction

Instructions are not created directly, but via a builder. The builder
both creates instructions and adds them to a basic block at the same
time. One way of getting instruction objects are from basic blocks.

Being derived from `llvm.core.User <llvm.core.User.html>`_, the
instruction is-a user, i.e., an instruction in turn uses other values.
The values an instruction uses are its operands. These may be accessed
using ``operands`` property from the
`llvm.core.User <llvm.core.User.html>`_ base.

The name of the instruction (like ``add``, ``mul`` etc) can be got via
the ``opcode_name`` property. The ``basic_block`` property gives the
basic block to which the instruction belongs to. Note that llvmpy does
not allow free-standing instruction objects (i.e., all instructions are
created contained within a basic block).

Classes of instructions can be got via the properties ``is_terminator``,
``is_binary_op``, ``is_shift`` etc. See below for the full list.

-  This will become a table of contents (this text will be scraped).
   {:toc}

llvm.core.Instruction
=====================

Base Class
----------

-  `llvm.core.User <llvm.core.User.html>`_

Properties
----------

``basic_block``
~~~~~~~~~~~~~~~

[read-only] The basic block to which this instruction belongs to.

``is_terminator``
~~~~~~~~~~~~~~~~~

[read-only] True if the instruction is a terminator instruction.

``is_binary_op``
~~~~~~~~~~~~~~~~

[read-only] True if the instruction is a binary operator.

``is_shift``
~~~~~~~~~~~~

[read-only] True if the instruction is a shift instruction.

``is_cast``
~~~~~~~~~~~

[read-only] True if the instruction is a cast instruction.

``is_logical_shift``
~~~~~~~~~~~~~~~~~~~~

[read-only] True if the instruction is a logical shift instruction.

``is_arithmetic_shift``
~~~~~~~~~~~~~~~~~~~~~~~

[read-only] True if the instruction is an arithmetic shift instruction.

``is_associative``
~~~~~~~~~~~~~~~~~~

[read-only] True if the instruction is associative.

``is_commutative``
~~~~~~~~~~~~~~~~~~

[read-only] True if the instruction is commutative.

``is_volatile``
~~~~~~~~~~~~~~~

[read-only] True if the instruction is a volatile load or store.

``opcode``
~~~~~~~~~~

[read-only] The numeric opcode value of the instruction. Do not rely on
the absolute value of this number, it may change with LLVM version.

``opcode_name``
~~~~~~~~~~~~~~~

[read-only] The name of the instruction, like ``add``, ``sub`` etc.

--------------

llvm.core.CallOrInvokeInstruction
=================================

The ``llvm.core.CallOrInvokeInstruction`` is a subclass of
``llvm.core.Instruction``, and represents either a ``call`` or an
``invoke`` instruction.

Base Class
----------

-  ``llvm.core.Instruction``

Properties
----------

``calling_convention`` Get or set the calling convention. See
`here <functions.html#callconv>`_ for possible values.

Methods
-------

``add_parameter_attribute(idx, attr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an attribute ``attr`` to the ``idx``-th argument. See
`here <llvm.core.Argument.html>`_ for possible values of ``attr``.

``remove_parameter_attribute(idx, attr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove an attribute ``attr`` from the ``idx``-th argument. See
`here <llvm.core.Argument.html>`_ for possible values of ``attr``.

``set_parameter_alignment(idx, align)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the alignment of the ``idx``-th argument to ``align``. ``align``
should be a power of two.

--------------

llvm.core.PHINode
=================

The ``llvm.core.PHINode`` is a subclass of ``llvm.core.Instruction``,
and represents the ``phi`` instruction. When created (using
``Builder.phi``) the phi node contains no incoming blocks (nor their
corresponding values). To add an incoming arc to the phi node, use the
``add_incoming`` method, which takes a source block
(`llvm.core.BasicBlock <llvm.core.BasicBlock.html>`_ object) and a value
(object of `llvm.core.Value <llvm.core.Value.html>`_ or of a class
derived from it) that the phi node will take on if control branches in
from that block.

Base Class
----------

-  ``llvm.core.Instruction``

Properties
----------

``incoming_count`` [read-only] The number of incoming arcs for this phi
node.

Methods
-------

``add_incoming(value, block)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add an incoming arc, from the
`llvm.core.BasicBlock <llvm.core.BasicBlock.html>`_ object ``block``,
with the corresponding value ``value``. ``value`` should be an object of
`llvm.core.Value <llvm.core.Value.html>`_ (or of a descendent class).

``get_incoming_value(idx)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the ``idx``-th incoming arc's value.

``get_incoming_block(idx)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the ``idx``-th incoming arc's block.

llvm.core.SwitchInstruction # {#switchinstr}
============================================

(TODO describe)

Base Class
----------

-  ``llvm.core.Instruction``

Methods
-------

``add_case(const, block)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Add another case to the switch statement. When the expression being
evaluated equals ``const``, then control branches to ``block``. Here
``const`` must be of type
`llvm.core.ConstantInt <llvm.core.Constant.html>`_.

--------------

llvm.core.CompareInstruction
============================

(TODO describe)

Base Class
----------

-  ``llvm.core.Instruction``

Properties
----------

``predicate``
~~~~~~~~~~~~~

[read-only]

The predicate of the compare instruction, one of the ``ICMP_*`` or
``FCMP_*`` constants.
