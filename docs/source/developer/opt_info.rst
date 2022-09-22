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
format that can direct Numba users on how to improve their code. For details on how to
use the processors, consult :doc:`performance tips <../user/performance-tips>`


Developing a Processor
----------------------

:class:`numba.core.opt_info.OptimizationProcessor` is the base class for processors.

When processors are used, Numba will first collect all the passes that are of interest to the processors using the
:meth:`numba.core.opt_info.OptimizationProcessor.filters` method. Each of the strings provided is a regular expression.
All of the expressions are combined into a a single large regular expression that is passed to LLVM.

Then, LLVM will compile the Numba-generated code. Numba uses multiple rounds of passes and remarks will be generated for
each round as a YAML file. Numba will read each YAML file and create an ordered list of parsed remarks. The YAML format
used by LLVM uses tags for different optimization events. These are converted to Python named tuples: ``Missed``,
``Passed``, ``Analysis``, ``AnalysisFPCommute``, ``AnalysisAliasing``, and ``Failure``.

Each processor's :meth:`numba.core.opt_info.OptimizationProcessor.process` method is then run with the parsed YAML data
and can produce pairs of strings and arbitrary data. The pairs will be used to populate a dictionary that will be
attached to the function's metadata as ``opt_info``.

For initial development, it can be useful to inspect the LLVM remarks output. The ``RawOptimizationRemarks`` can be used
to extract remarks without modification for development purposes. The constructor takes the names of the passes that
should be included.