.. _cuda-jit-module:

============================================
Automatic module jitting with ``jit_module``
============================================

The :ref:`jit_module function <jit-module>` is supported by the CUDA target, as
:func:`numba.cuda.jit_module`.

.. autofunction:: numba.cuda.jit_module
   :noindex:


Example Usage
=============

It can be useful to use the ``jit_module`` decorator twice in a module; once
for device functions, and once for kernels that call the device functions.
Following the second call to ``jit_module``, host functions can be added - these
may be used to call kernels jitted earlier.

The following example demonstrates this pattern.

.. literalinclude:: ../../../numba/cuda/tests/doc_examples/test_jit_module.py
   :language: python
   :caption: from ``test_multiple_jit_module`` of ``numba/cuda/tests/doc_example/test_jit_module.py``
   :start-after: magictoken.ex_multiple_jit_module.begin
   :end-before: magictoken.ex_multiple_jit_module.end
   :dedent: 8
   :linenos:


