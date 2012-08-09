+-----------------+
| layout: page    |
+-----------------+
| title: Values   |
+-----------------+

`llvm.core.Value <llvm.core.Value.html>`_ is the base class of all
values computed by a program that may be used as operands to other
values. A value has a type associated with it (an object of
`llvm.core.Type <types.html>`_).

The class hierarchy is:

::

    Value
      User
        Constant
          ConstantExpr
          ConstantAggregateZero
          ConstantInt
          ConstantFP
          ConstantArray
          ConstantStruct
          ConstantVector
          ConstantPointerNull
          UndefValue
          GlobalValue
            GlobalVariable
            Function
        Instruction
          CallOrInvokeInstruction
          PHINode
          SwitchInstruction
          CompareInstruction
      Argument
      BasicBlock

The `Value <llvm.core.Value.html>`_ class is abstract, it's not meant to
be instantiated. `User <llvm.core.User.html>`_ is a
`Value <llvm.core.Value.html>`_ that in turn uses (i.e., can refer to)
other values (for e.g., a constant expression 1+2 refers to two constant
values 1 and 2).

`Constant <llvm.core.Constant.html>`_-s represent constants that appear
within code or as initializers of globals. They are constructed using
static methods of `Constant <llvm.core.Constant.html>`_. Various types
of constants are represented by various subclasses of
`Constant <llvm.core.Constant.html>`_. However, most of them are empty
and do not provide any additional attributes or methods over
`Constant <llvm.core.Constant.html>`_.

The `Function <functions.html>`_ object represents an instance of a
function type. Such objects contain
`Argument <llvm.core.Argument.html>`_ objects, which represent the
actual, local-variable-like arguments of the function (not to be
confused with the arguments returned by a function *type* object --
these represent the *type* of the arguments).

The various `Instruction <llvm.core.Instruction.html>`_-s are created by
the `Builder <llvm.core.Builder.html>`_ class. Most instructions are
represented by `Instruction <llvm.core.Instruction.html>`_ itself, but
there are a few subclasses that represent interesting instructions.

`Value <llvm.core.Value.html>`_ objects have a type (read-only), and a
name (read-write).

**Related Links** `functions <functions.html>`_,
`comparision <comparision.html>`_,
`llvm.core.Value <llvm.core.Value.html>`_,
`llvm.core.User <llvm.core.User.html>`_,
`llvm.core.Constant <llvm.core.Constant.html>`_,
`llvm.core.GlobalValue <llvm.core.GlobalValue.html>`_,
`llvm.core.GlobalVariable <llvm.core.GlobalVariable.html>`_,
`llvm.core.Argument <llvm.core.Argument.html>`_,
`llvm.core.Instruction <llvm.core.Instruction.html>`_,
`llvm.core.Builder <llvm.core.Builder.html>`_,
`llvm.core.BasicBlock <llvm.core.BasicBlock.html>`_
