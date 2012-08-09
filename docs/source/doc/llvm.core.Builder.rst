+------------------------------+
| layout: page                 |
+------------------------------+
| title: Builder (llvm.core)   |
+------------------------------+

The ``Builder`` class corresponds to the
`IRBuilder <http://llvm.org/docs/doxygen/html/classllvm_1_1IRBuilder.html>`_
in C++ llvm. It provides an uniform API to populating
`BasicBlocks <llvm.core.BasicBlock.html>`_. Most of the methods in
``Builder`` correspond to the instructions in the LLVM IR. See `LLVM
documentation <http://llvm.org/docs/LangRef.html>`_ for detail. These
methods have the ``name`` argument for overiding the name of the result
variable. When it is an empty string (default value), LLVM will set a
numeric ID for the result variable.

llvm.core.Builder
=================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Static Factor Method
--------------------

``new(basic_block)``
~~~~~~~~~~~~~~~~~~~~

Create an instance of ``Builder`` at
`BasicBlock <llvm.core.BasicBlock.html>`_.

Methods
-------

``add(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs+rhs`` for integer values only.

``alloca(self, ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that allocates stack memory for a value of type
``ty``.

``alloca_array(self, ty, size, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that allocates stack memory for a ``size``
elements array of type ``ty``.

``and_(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs & rhs``.

``ashr(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs >> rhs`` using arithmetic
shift.

``bitcast(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that cast ``value`` to type ``dest_ty``.

``branch(self, bblk)``
~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that branch to basicblock ``bblk``.

``call(self, fn, args, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that call function ``fn`` with a iterable of
arguments ``args``.

``cbranch(self, if_value, then_blk, else_blk)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that conditionally branch base on the predicate
``if_value``. If ``if_value`` is ``True``, branch to ``then_blk``;
Otherwise, branch to ``else_blk``.

``extract_element(self, vec_val, idx_val, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that extracts an element from a value ``vec_val``
of `llvm.core.VectorType <llvm.core.VectorType.html>`_ at index
``idx_val``.

``extract_value(self, retval, idx, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that extracts an element from an aggregate value
``retval`` at index ``idx``.

``fadd(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs + rhs`` for floating-point
values.

``fcmp(self, rpred, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that compares ``lhs`` and ``rhs`` using the
comparision operation defined by ``rpred``. See
`here <comparision.html#fcmp>`_ for a list of comparators.

``fdiv(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs / rhs`` for floating-point
values.

``fmul(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs * rhs`` for floating-point
values.

``fpext(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that extends ``value`` to a float type
``dest_ty``.

``fptosi(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that converts a floating-point value ``value`` to
a signed integer type ``dest_ty``.

``fptoui(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that converts a floating-point value ``value`` to
an unsigned integer type ``dest_ty``.

``fptrunc(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that truncates a floating-point value ``value`` to
a float type ``dest_ty``.

``free(self, ptr)``
~~~~~~~~~~~~~~~~~~~

Insert an instruction that call performs heap deallocation on pointer
``ptr``.

``frem(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs % rhs`` for floating-point
values.

``fsub(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs - rhs`` for floating-point
values.

``gep(self, ptr, indices, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `GEP <http://llvm.org/docs/LangRef.html#i_getelementptr>`_.

``getresult(self, retval, idx, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

same as ``extract_value``.

``icmp(self, ipred, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that compares ``lhs`` and ``rhs`` using the
comparision operation defined by ``ipred``. See
`here <comparision.html#icmp>`_ for a list of comparators.

``insert_element(self, vec_val, elt_val, idx_val, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that inserts a value ``elt_val`` into ``vec_val``
of `llvm.core.VectorType <llvm.core.VectorType.html>`_ at index
``idx_val``.

``inttoptr(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that converts an integer ``value`` to pointer
``dest_ty``.

``invoke(self, func, args, then_blk, catch_blk, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See `invoke <http://llvm.org/docs/LangRef.html#i_invoke>`_

``load(self, ptr, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that loads a value at the memory pointed by
``ptr``.

``lshr(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs >> rhs`` using logical shift.

``malloc(self, ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that allocates heap memory of type ``ty``. The
instruction returns a pointer that points to a value of type ``ty``.

``malloc_array(self, ty, size, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to ``malloc`` but allocates an array of ``size`` elements.

``mul(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs * rhs`` for integer types.

``neg(self, val, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``0 - val``.

``not_(self, val, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes an one's complement of ``val``.

``or_(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs | rhs``.

``phi(self, ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a PHI node of type ``ty``.

``position_at_beginning(self, bblk)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Position the builder at the beginning of the given block. Next
instruction inserted will be first one in the block.

``position_at_end(self, bblk)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Position the builder at the end of the given block. Next instruction
inserted will be last one in the block.

``position_before(self, instr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Position the builder before the given instruction. The instruction can
belong to a basic block other than the current one.

``ptrtoint(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that converts a pointer to an integer ``value`` of
type ``dest_ty``.

``ret(self, value)``
~~~~~~~~~~~~~~~~~~~~

Insert an instruction that returns ``value``.

``ret_many(self, values)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that returns ``values`` which is an iterable of
`llvm.core.Value <llvm.core.Value.html>`_.

``ret_void(self)``
~~~~~~~~~~~~~~~~~~

Insert an instruction that returns nothing (void).

``sdiv(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs / rhs`` for signed integers.

``select(self, cond, then_value, else_value, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``cond ? then_value : else_value``.

``sext(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that sign extends an integer ``value`` to type
``dest_ty``.

``shl(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs << rhs``.

``shuffle_vector(self, vecA, vecB, mask, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that performs a vector shuffle base on the two
vectors -- ``vecA`` and ``vecB``, base on a bit mask ``mask``. The mask
must be a constant.

See `LLVM document <http://llvm.org/docs/LangRef.html#i_shufflevector>`_
for detail.

``sitofp(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that converts a signed integer ``value`` to a
floating-point type ``dest_ty``.

``srem(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs % rhs`` for signed integers.

``store(self, value, ptr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that stores ``value`` into the memory pointed by
``ptr``.

``sub(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs - rhs``.

``switch(self, value, else_blk, n=10)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that transfer control flow depending on the
``value``. ``else_blk`` is the default case. ``n`` sets the number of
additional cases.

This method returns an instance of
`SwitchInstruction <llvm.core.Instruction.html#switchinstr>`_ for adding
cases to the switch.

``trunc(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that truncates an integer ``value`` to the
destination integer type ``dest_ty``.

``udiv(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs / rhs`` for unsigned integers.

``uitofp(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that converts an unsigned integer ``value`` to a
floating-point type ``dest_ty``.

``unreachable(self)``
~~~~~~~~~~~~~~~~~~~~~

Insert an unreachabe instruction, which has no defined semantics. See
`LLVM document <http://llvm.org/docs/LangRef.html#i_unreachable>`_ for
detail.

``urem(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs % rhs`` for unsigned integers.

``vaarg(self, list_val, ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is used to access variable arguments given as ``list_val`` of type
``ty``. see `LLVM
document <http://llvm.org/docs/LangRef.html#int_varargs>`_ about
variable argument intrinsics.

``xor(self, lhs, rhs, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that computes ``lhs xor rhs``.

``zext(self, value, dest_ty, name='')``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Insert an instruction that zero extends ``value`` to type ``dest_ty``.

Properties
----------

``basic_block``
~~~~~~~~~~~~~~~

The `BasicBlock <llvm.core.BasicBlock.html>`_ where the builder is
positioned.

``block``
~~~~~~~~~

Deprecated. Same as ``basic_block``
