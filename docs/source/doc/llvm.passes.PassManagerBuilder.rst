+-------------------------------------------+
| layout: page                              |
+-------------------------------------------+
| title: PassManagerBuilder (llvm.passes)   |
+-------------------------------------------+

llvm.passes.PassManagerBuilder
==============================

Provide a simple API to populate pass managers for language like C/C++.
Refer to `LLVM API
Documentation <http://llvm.org/docs/doxygen/html/classllvm_1_1PassManagerBuilder.html>`_
for detail.

Methods
-------

``populate(self, pm)``
~~~~~~~~~~~~~~~~~~~~~~

Populate a `FunctionPassManager <llvm.passes.FunctionPassManager.html>`_
or `PassManager <llvm.passes.PassManager.html>`_ given as ``pm``.

``use_inliner_with_threshold(self, threshold)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an inliner pass with the given ``threshold``.

Properties
----------

The following properties can be overriden to customize how pass managers
are populated.

``disable_simplify_lib_calls``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Boolean. Default is ``False``.

``disable_unit_at_a_time``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Boolean. Default is ``False``.

``disable_unroll_loops``
~~~~~~~~~~~~~~~~~~~~~~~~

Boolean. Default is ``False``.

``opt_level``
~~~~~~~~~~~~~

Default is ``2``. Valid values are 0-3. Corresponds to O0, O1, O2, O3 as
in C/C++ optimization options.

``size_level``
~~~~~~~~~~~~~~

Default is ``0``.

``vectorize``
~~~~~~~~~~~~~

Default is ``False``.

Static Factory Methods
----------------------

``new()``
~~~~~~~~~

Creates a new ``PassManagerBuilder`` instance.
