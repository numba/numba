.. _arch-pipeline:

========================
Customizing the Compiler
========================

.. warning:: The custom pipeline feature is for expert use only.  Modifying
             the compiler behavior can invalidate internal assumptions in the
             numba source code.


For library developers looking for a way to extend or modify the compiler
behavior, you can do so by defining a custom compiler by inheriting from
``numba.compiler.CompilerBase``.  The default Numba compiler is defined
as ``numba.compiler.Compiler``, implementing the ``.define_pipelines()``
method, which adds the *nopython-mode*, *object-mode* and *interpreted-mode*
pipelines. For convenience these three pipelines are defined in
``numba.compiler.DefaultPassBuilder`` by the methods:

* ``.define_nopython_pipeline()``
* ``.define_objectmode_pipeline()``
* ``.define_interpreted_pipeline()``

respectively.

To use a custom subclass of ``CompilerBase``, supply it as the
``pipeline_class`` keyword argument to the ``@jit`` and ``@generated_jit``
decorators.  By doing so, the effect of the custom pipeline is limited to the
function being decorated.

Implementing a compiler pass
----------------------------

Numba makes it possible to implement a new compiler pass and does so through the
use of an API similar to that of LLVM. The following demonstrates the basic
process involved.


Compiler pass classes
#####################

All passes must inherit from ``numba.compiler_machinery.CompilerPass``, commonly
used subclasses are:

* ``numba.compiler_machinery.FunctionPass`` for describing a pass that operates
  on a function-at-once level and may mutate the IR state.
* ``numba.compiler_machinery.AnalysisPass`` for describing a pass that performs
  analysis only.
* ``numba.compiler_machinery.LoweringPass`` for describing a pass that performs
  lowering only.

In this example a new compiler pass will be implemented that will rewrite all
``ir.Const(x)`` nodes, where ``x`` is a subclass of ``numbers.Number``, such
that the value of x is incremented by one. There is no use for this pass other
than to serve as a pedagogical vehicle!

The ``numba.compiler_machinery.FunctionPass`` is appropriate for the suggested
pass behavior and so is the base class of the new pass. Further, a ``run_pass``
method is defined to do the work (this method is abstract, all compiler passes
must implement it).

First the new class:

.. literalinclude:: compiler_pass_example.py
   :language: python
   :dedent: 4
   :start-after: magictoken.ex_compiler_pass.begin
   :end-before: magictoken.ex_compiler_pass.end


Note also that the class must be registered with Numba's compiler machinery
using ``@register_pass``. This in part is to allow the declaration of whether
the pass mutates the control flow graph and whether it is an analysis only pass.

Next, define a new compiler based on the existing
``numba.compiler.CompilerBase``. The compiler pipeline is defined through the
use of an existing pipeline and the new pass declared above is added to be run
after the ``IRProcessing`` pass.


.. literalinclude:: compiler_pass_example.py
   :language: python
   :dedent: 4
   :start-after: magictoken.ex_compiler_defn.begin
   :end-before: magictoken.ex_compiler_defn.end

Finally update the ``@njit`` decorator at the call site to make use of the newly
defined compilation pipeline.

.. literalinclude:: compiler_pass_example.py
   :language: python
   :dedent: 4
   :start-after: magictoken.ex_compiler_call.begin
   :end-before: magictoken.ex_compiler_call.end

Debugging compiler passes
-------------------------

Observing IR Changes
####################

It is often useful to be able to see the changes a pass makes to the IR. Numba
conveniently permits this through the use of the environment variable
:envvar:`NUMBA_DEBUG_PRINT_AFTER`. In the case of the above pass, running the
example code with ``NUMBA_DEBUG_PRINT_AFTER="ir_processing,consts_add_one"``
gives:


.. code-block:: none
    :emphasize-lines: 4, 7, 24, 27

    ----------------------------nopython: ir_processing-----------------------------
    label 0:
        x = arg(0, name=x)                       ['x']
        $const0.1 = const(int, 10)               ['$const0.1']
        a = $const0.1                            ['$const0.1', 'a']
        del $const0.1                            []
        $const0.2 = const(float, 20.2)           ['$const0.2']
        b = $const0.2                            ['$const0.2', 'b']
        del $const0.2                            []
        $0.5 = x + a                             ['$0.5', 'a', 'x']
        del x                                    []
        del a                                    []
        $0.7 = $0.5 + b                          ['$0.5', '$0.7', 'b']
        del b                                    []
        del $0.5                                 []
        c = $0.7                                 ['$0.7', 'c']
        del $0.7                                 []
        $0.9 = cast(value=c)                     ['$0.9', 'c']
        del c                                    []
        return $0.9                              ['$0.9']
    ----------------------------nopython: consts_add_one----------------------------
    label 0:
        x = arg(0, name=x)                       ['x']
        $const0.1 = const(int, 11)               ['$const0.1']
        a = $const0.1                            ['$const0.1', 'a']
        del $const0.1                            []
        $const0.2 = const(float, 21.2)           ['$const0.2']
        b = $const0.2                            ['$const0.2', 'b']
        del $const0.2                            []
        $0.5 = x + a                             ['$0.5', 'a', 'x']
        del x                                    []
        del a                                    []
        $0.7 = $0.5 + b                          ['$0.5', '$0.7', 'b']
        del b                                    []
        del $0.5                                 []
        c = $0.7                                 ['$0.7', 'c']
        del $0.7                                 []
        $0.9 = cast(value=c)                     ['$0.9', 'c']
        del c                                    []
        return $0.9                              ['$0.9']

Note the change in the values in the ``const`` nodes.

Pass execution times
####################

Numba has built-in support for timing all compiler passes, the execution times
are stored in the metadata associated with a compilation result. This
demonstrates one way of accessing this information based on the previously
defined function, ``foo``:

.. literalinclude:: compiler_pass_example.py
   :language: python
   :dedent: 4
   :start-after: magictoken.ex_compiler_timings.begin
   :end-before: magictoken.ex_compiler_timings.end

the output of which is, for example::

    pass_timings(init=1.914000677061267e-06, run=4.308700044930447e-05, finalize=1.7400006981915794e-06)

this displaying the pass initialization, run and finalization times in seconds.
