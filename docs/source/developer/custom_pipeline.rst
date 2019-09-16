.. _arch-pipeline:

=========================
Notes on Custom Pipelines
=========================

.. warning:: The custom pipeline feature is for expert use only.  Modifying
             the compiler behavior can invalidate internal assumptions in the
             numba source code.


For library developers looking for a way to extend or modify the compiler
behavior, you can do so by defining a custom compiler by inheriting from
``numba.compiler.CompilerBase``.  The default Numba compiler is defined
as ``numba.compiler.Compiler``, implementing the ``.define_pipelines()``
method, which adds the *nopython-mode*, *object-mode* and *interpreted-mode*
pipelines.  These three pipelines are defined in
``numba.compiler.DefaultPassBuilder`` by the methods
``.define_nopython_pipeline``, ``.define_objectmode_pipeline`` and
``.define_interpreted_pipeline``, respectively.

To use a custom subclass of ``CompilerBase``, supply it as the
``pipeline_class`` keyword argument to the ``@jit`` and ``@generated_jit``
decorators.  By doing so, the effect of the custom pipeline is limited to the
function being decorated.
