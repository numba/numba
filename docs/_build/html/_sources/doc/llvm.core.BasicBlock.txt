+---------------------------------+
| layout: page                    |
+---------------------------------+
| title: BasicBlock (llvm.core)   |
+---------------------------------+

A basicblock is a list of instructions. A wellformed basicblock should
end with a terminator. ``Function.verify()`` will verify that. A
terminator is either a branch instruction or return instruction. It is
not possible to have instructions after a branch or return instruction.

llvm.core.BasicBlock
====================

Base Class
----------

-  `llvm.core.Value <llvm.core.Value.html>`_

Methods
-------

``delete(self)``
~~~~~~~~~~~~~~~~

Delete this basicblock from the function (``self.function``).

``insert_before(self, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO

Proporties
----------

``function``
~~~~~~~~~~~~

The parent function of this basicblock.

``instructions``
~~~~~~~~~~~~~~~~

A list of instructions in this basicblock.
