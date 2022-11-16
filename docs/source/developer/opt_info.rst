=================================
Notes on Optimization Information
=================================

.. warning:: All features and APIs described in this page are in-development and
             may change at any time without deprecation notices being issued.


LLVM Optimization Remarks
=========================

During compilation, LLVM passes can emit optimization remarks that provide information
on how well each pass was able to detect and rewrite code to be more efficient. Numba
provides `optimization processors` to extract these remarks and convert them into a
format that can direct Numba users on how to improve their code.

Using Processors
----------------

Numba uses LLVM to create optimized machine code and LLVM provides information
about which optimizations it tried and which succeeded. This information can be
accessed on Numba-compiled functions. First, optimization processors must be
added. Processors can be enabled globally::

    import numba.core.opt_info
    numba.core.opt_info.register_processor(numba.core.opt_info.LoopVectorizationDetector())

or on an individual function::

    from numba import njit
    import numba.core.opt_info

    @njit(opt_info=[numba.core.opt_info.SuperWorldLevelParallelismDetector()])
    def my_func(a, b, c):
       return a * b + b * c + a * c

After using your function, the optimization information will be available in
the function's metadata::

    all_opt_info = {signature: metadata['opt_info'] for (signature, metadata)
        in my_func.get_metadata().items()}

If the function was called with different types, there may be multiple entries
since the optimizations for each type can be different.

The current processors available are:

- ``LoopVectorizationDetector()`` - Indicates whether a loop vectorization
  happened.  Vectorization replaces loops with CPU instructions that can do
  operations in parallel. The resulting loop will be much faster. Not all loops
  can be vectorized as the set of CPU instructions is limited to "pure
  arithmetic" operations.

  The output of this processor will be a dictionary with a (file, line,
  column) tuple and a true if the loop was vectorized; false otherwise.
- ``SuperWorldLevelParallelismDetector()`` - Determines locations where the `SLP
  detector <https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer>`_ found a
  way to turn scalar operations into vector operations.
  CPUs have some operations that are meant for doing arithmetic operations
  over vectors quickly. Sometimes, even though code is being done over scalar
  values, it has a structure that looks like a vector, so it can be more
  efficient to convert the data into a vector and use the special operations.

  The output of this processor is a set of (file, line, column) tuples where
  scalar operations were successfully converted to a faster vector operation.

Developing a Processor
----------------------

:class:`numba.core.opt_info.OptimizationProcessor` is the base class for processors.

When processors are used, Numba will first collect all the passes that are of interest to the processors using the
:meth:`numba.core.opt_info.OptimizationProcessor.filters` method. Each of the strings provided is a regular expression.
All of the expressions are combined into a a single large regular expression that is passed to LLVM.

Then, LLVM will compile the Numba-generated code. Numba uses multiple rounds of
passes and remarks will be generated for each round as a YAML file. Numba will
read each YAML file and create a dictionary list of parsed remarks. The rounds
are `"cheap"`, then `"full"`. The YAML format used by LLVM uses tags for
different optimization events. These are converted to Python named tuples:
``Missed``, ``Passed``, ``Analysis``, ``AnalysisFPCommute``,
``AnalysisAliasing``, and ``Failure``.

Each processor's :meth:`numba.core.opt_info.OptimizationProcessor.process` method is then run with the parsed YAML data
and can produce pairs of strings and arbitrary data. The pairs will be used to populate a dictionary that will be
attached to the function's metadata as ``opt_info``.

For initial development, it can be useful to inspect the LLVM remarks output. The ``RawOptimizationRemarks`` can be used
to extract remarks without modification for development purposes. The constructor takes the names of the passes that
should be included.
