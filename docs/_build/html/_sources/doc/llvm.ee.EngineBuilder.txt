+----------------------------------+
| layout: page                     |
+----------------------------------+
| title: EngineBuilder (llvm.ee)   |
+----------------------------------+

llvm.ee.EngineBuilder
=====================

A convenient class for building
`llvm.ee.ExecutionEngine <llvm.ee.ExecutionEngine.html>`_. Each
``EngineBuilder`` instance can only create one ``ExecutionEngine``.

Methods
-------

``create(self)``
~~~~~~~~~~~~~~~~

Create and return a new
`ExecutionEngine <llvm.ee.ExecutionEngine.html>`_ instance.

Raise ``llvm.LLVMException`` if the builder cannot create an
``ExecutionEngine`` base on the given configuration.

``force_interpreter(self)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Force the output the output ``ExecutionEngine`` to be an LLVM IR
interpreter.

``force_jit(self)``
~~~~~~~~~~~~~~~~~~~

Force the output the output ``ExecutionEngine`` to be a JIT engine.

``opt(self, level)``
~~~~~~~~~~~~~~~~~~~~

Set the code generation optimization level for a JIT engine. Valid value
of ``level`` is 0-3, inclusive. The default setting is 2. To use vector
instructions, such as SSE on Intel processors, ``level`` must be 3
(aggressive).

Static Factory Methods
----------------------

``new(module)``
~~~~~~~~~~~~~~~

Create a new EngineBuilder. ``module`` must be a
`llvm.core.Module <llvm.core.Module.html>`_ instance. Its ownership is
transferred to the resulting
`ExecutionEngine <llvm.ee.ExecutionEngine.html>`_. Therefore, it is
impossible to create more than one ``ExecutionEngine`` with a single
``EngineBuilder``
