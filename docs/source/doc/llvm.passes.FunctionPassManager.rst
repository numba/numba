+--------------------------------------------+
| layout: page                               |
+--------------------------------------------+
| title: FunctionPassManager (llvm.passes)   |
+--------------------------------------------+

llvm.passes.FunctionPassManager
===============================

Base Classes
------------

-  `llvm.passes.PassManager <llvm.passes.PassManager.html>`_

Methods
-------

``finalize(self)``
~~~~~~~~~~~~~~~~~~

Finalizes all associated function passes in the LLVM system.

Beware that this destroys all associated passes even if another pass
manager is using those passes. This may result is a segfault.

``initialize(self)``
~~~~~~~~~~~~~~~~~~~~

Initializes all associated function passes in the LLVM system.

``run(self, fn)``
~~~~~~~~~~~~~~~~~

Run all passes on the given function ``fn``.

Static Factory Methods
----------------------

``new(module)``
~~~~~~~~~~~~~~~

Create a ``FunctionPassManager`` instance for a given ``module``.
