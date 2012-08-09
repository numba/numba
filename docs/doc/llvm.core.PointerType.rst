+----------------------------------+
| layout: page                     |
+----------------------------------+
| title: PointerType (llvm.core)   |
+----------------------------------+

llvm.core.PointerType
=====================

Base Class
----------

-  `llvm.core.Type <llvm.core.Type.html>`_

Properties
----------

``address_space``
~~~~~~~~~~~~~~~~~

[read-only]

The address space of the pointer.

``pointee``
~~~~~~~~~~~

[read-only]

A `Type <llvm.core.Type.html>`_ object representing the type of the
value pointed to.
