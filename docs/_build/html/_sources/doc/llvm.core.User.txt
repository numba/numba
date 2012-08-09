+---------------------------+
| layout: page              |
+---------------------------+
| title: User (llvm.core)   |
+---------------------------+

``User``-s are values that refer to other values. The values so refered
can be retrived by the properties of ``User``. This is the reverse of
the ``Value.uses``. Together these can be used to traverse the use-def
chains of the SSA.

--------------

llvm.core.User # {#user}
========================

Base Class
----------

-  `llvm.core.Value <llvm.core.Value.html>`_

Properties
----------

``operands``
~~~~~~~~~~~~

[read-only]

The list of operands (values, of type
`llvm.core.Value <llvm.core.Value.html>`_) that this value refers to.

``operand_count``
~~~~~~~~~~~~~~~~~

[read-only]

The number of operands that this value referes to. Same as
``len(uses.operands)`` but faster if you just want the count.
