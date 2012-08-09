+-------------------------------+
| layout: page                  |
+-------------------------------+
| title: Function (llvm.core)   |
+-------------------------------+

llvm.core.Function
==================

-  This will become a table of contents (this text will be scraped).
   {:toc}

Base Class
----------

-  `llvm.core.GlobalValue <llvm.core.GlobalValue.html>`_

Static Constructors
-------------------

``new(module_obj, func_ty, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a function named ``name`` of type ``func_ty`` in the module
``module_obj`` and return a ``Function`` object that represents it.

``get(module_obj, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~

Return a ``Function`` object to represent the function named ``name`` in
the module ``module_obj`` or raise ``LLVMException`` if such a function
does not exist.

``get_or_insert(module_obj, func_ty, name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to ``get``, except that if the function does not exist it is
added first, as though with ``new``.

``intrinsic(module_obj, intrinsic_id, types)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and return a ``Function`` object that refers to an intrinsic
function, as described `here <functions.html#intrinsic>`_.

Properties
----------

``calling_convention``
~~~~~~~~~~~~~~~~~~~~~~

The calling convention for the function, as listed
`here <functions.html#callconv>`_.

``collector``
~~~~~~~~~~~~~

A string holding the name of the garbage collection algorithm. See `LLVM
docs <http://www.llvm.org/docs/LangRef.html#gc>`_.

``does_not_throw``
~~~~~~~~~~~~~~~~~~

Setting to True sets the ``ATTR_NO_UNWIND`` attribute, False removes it.
Shortcut to using ``f.add_attribute(ATTR_NO_UNWIND)`` and
``f.remove_attribute(ATTR_NO_UNWIND)``.

``args``
~~~~~~~~

[read-only]

List of `llvm.core.Argument <llvm.core.Argument.html>`_ objects
representing the formal arguments of the function.

``basic_block_count``
~~~~~~~~~~~~~~~~~~~~~

[read-only]

Number of basic blocks belonging to this function. Same as
``len(f.basic_blocks)`` but faster if you just want the count.

``entry_basic_block``
~~~~~~~~~~~~~~~~~~~~~

[read-only]

The `llvm.core.BasicBlock <llvm.core.BasicBlock.html>`_ object
representing the entry basic block for this function, or ``None`` if
there are no basic blocks.

``basic_blocks``
~~~~~~~~~~~~~~~~

[read-only]

List of `llvm.core.BasicBlock <llvm.core.BasicBlock.html>`_ objects
representing the basic blocks belonging to this function.

``intrinsic_id``
~~~~~~~~~~~~~~~~

[read-only]

Returns the ID of the intrinsic if this object represents an intrinsic
instruction. Otherwise 0.

Methods
-------

``delete()``
~~~~~~~~~~~~

Deletes the function from it's module. Do not hold any references to
this object after calling ``delete`` on it.

``append_basic_block(name)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a new basic block named ``name``, and return a corresponding
`llvm.core.BasicBlock <llvm.core.BasicBlock.html>`_ object. Note that if
this is not the entry basic block, you'll have to add appropriate branch
instructions from other basic blocks yourself.

``add_attribute(attr)``
~~~~~~~~~~~~~~~~~~~~~~~

Add an attribute ``attr`` to the function, from the set listed above.

``remove_attribute(attr)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove the attribute ``attr`` of the function.

``viewCFG()``
~~~~~~~~~~~~~

Displays the control flow graph using the GraphViz tool.

``viewCFGOnly()``
~~~~~~~~~~~~~~~~~

Displays the control flow graph using the GraphViz tool, but omitting
function bodies.

``verify()``
~~~~~~~~~~~~

Verifies the function. See `LLVM
docs <http://llvm.org/docs/Passes.html#verify>`_.
