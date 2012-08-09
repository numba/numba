+------------------------------------+
| layout: page                       |
+------------------------------------+
| title: PassManager (llvm.passes)   |
+------------------------------------+

llvm.passes.PassManager
=======================

Methods
-------

``add(self, tgt_data_or_pass_id)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a pass by its ID. A pass IDs are defined as ``PASS_*``.

``run(self, module)``
~~~~~~~~~~~~~~~~~~~~~

Run all passes on the given ``module``.

Static Factory Methods
----------------------

``new()``
~~~~~~~~~

Creates a new ``PassManager`` instance.
