+------------------------------------+
| layout: page                       |
+------------------------------------+
| title: ExecutionEngine (llvm.ee)   |
+------------------------------------+

llvm.ee.ExecutionEngine
=======================

Methods
-------

``add_module(self, module)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add a new module to the ExecutionEngine. The ownership is of ``module``
is transferred. When the ``ExecutionEngine`` is destroyed, the module is
destroyed.

``free_machine_code_for(self, fn)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Release memory used for the machine code generated for the function
``fn``.

``get_pointer_to_function(self, fn)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obtain the pointer to the function ``fn``. This forces the
ExecutionEngine to generate the machine code in lazy mode.

If ``fn`` is not defined, ``ExecutionEngine`` will lookup the symbol
through ``dlsym``.

The returned function pointer can be wrapped as a ``ctypes`` function.

``remove_module(self, module)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Remove the ``module``.

``run_function(self, fn, args)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute the function ``fn`` with an iterable of arguments ``args`` which
are of ``GenericValue``. This method returns whatever that is returned
by ``fn`` as a ``GenericValue``.

``run_static_ctors(self)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

``run_static_dtors(self)``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Properties
----------

``target_data``
~~~~~~~~~~~~~~~

Access the `TargetData <llvm.ee.TargetData.html>`_ instance associated
with the ``ExecutionEngine``.
