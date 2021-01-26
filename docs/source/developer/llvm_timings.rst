.. _developer-llvm-timings:

====================
Notes on timing LLVM
====================


Getting LLVM Pass Timings
-------------------------

The dispatcher stores LLVM pass timings in the dispatcher object metadata under
the ``llvm_pass_timings`` key when :envvar:`NUMBA_LLVM_PASS_TIMINGS` is
enabled or ``numba.config.LLVM_PASS_TIMINGS`` is set to truthy.
The timings information contains details on how much time
has been spent in each pass. The pass timings are also grouped by their purpose.
For example, there will be pass timings for function-level pre-optimizations,
module-level optimizations, and object code generation.


Code Example
~~~~~~~~~~~~

.. literalinclude:: ../../../numba/tests/doc_examples/test_llvm_pass_timings.py
   :language: python
   :caption: from ``test_pass_timings`` of ``numba/tests/doc_examples/test_llvm_pass_timings.py``
   :start-after: magictoken.ex_llvm_pass_timings.begin
   :end-before: magictoken.ex_llvm_pass_timings.end
   :dedent: 16
   :linenos:

Example output:

.. code-block:: text

  Printing pass timings for JITCodeLibrary('DocsLLVMPassTimings.test_pass_timings.<locals>.foo')
  Total time: 0.0376
  == #0 Function passes on '_ZN5numba5tests12doc_examples22test_llvm_pass_timings19DocsLLVMPassTimings17test_pass_timings12$3clocals$3e7foo$241Ex'
  Percent: 4.8%
  Total 0.0018s
  Top timings:
    0.0015s ( 81.6%) SROA #3
    0.0002s (  9.3%) Early CSE #2
    0.0001s (  4.0%) Simplify the CFG #9
    0.0000s (  1.5%) Prune NRT refops #4
    0.0000s (  1.1%) Post-Dominator Tree Construction #5
  == #1 Function passes on '_ZN7cpython5numba5tests12doc_examples22test_llvm_pass_timings19DocsLLVMPassTimings17test_pass_timings12$3clocals$3e7foo$241Ex'
  Percent: 0.8%
  Total 0.0003s
  Top timings:
    0.0001s ( 30.4%) Simplify the CFG #10
    0.0001s ( 24.1%) Early CSE #3
    0.0001s ( 17.8%) SROA #4
    0.0000s (  8.8%) Prune NRT refops #5
    0.0000s (  5.6%) Post-Dominator Tree Construction #6
  == #2 Function passes on 'cfunc._ZN5numba5tests12doc_examples22test_llvm_pass_timings19DocsLLVMPassTimings17test_pass_timings12$3clocals$3e7foo$241Ex'
  Percent: 0.5%
  Total 0.0002s
  Top timings:
    0.0001s ( 27.7%) Early CSE #4
    0.0001s ( 26.8%) Simplify the CFG #11
    0.0000s ( 13.8%) Prune NRT refops #6
    0.0000s (  7.4%) Post-Dominator Tree Construction #7
    0.0000s (  6.7%) Dominator Tree Construction #29
  == #3 Module passes (cheap optimization for refprune)
  Percent: 3.7%
  Total 0.0014s
  Top timings:
    0.0007s ( 52.0%) Combine redundant instructions
    0.0001s (  5.4%) Function Integration/Inlining
    0.0001s (  4.9%) Prune NRT refops #2
    0.0001s (  4.8%) Natural Loop Information
    0.0001s (  4.6%) Post-Dominator Tree Construction #2
  == #4 Module passes (full optimization)
  Percent: 43.9%
  Total 0.0165s
  Top timings:
    0.0032s ( 19.5%) Combine redundant instructions #9
    0.0022s ( 13.5%) Combine redundant instructions #7
    0.0010s (  6.1%) Induction Variable Simplification
    0.0008s (  4.8%) Unroll loops #2
    0.0007s (  4.5%) Loop Vectorization
  == #5 Finalize object
  Percent: 46.3%
  Total 0.0174s
  Top timings:
    0.0060s ( 34.6%) X86 DAG->DAG Instruction Selection #2
    0.0019s ( 11.0%) Greedy Register Allocator #2
    0.0013s (  7.4%) Machine Instruction Scheduler #2
    0.0012s (  7.1%) Loop Strength Reduction
    0.0004s (  2.3%) Induction Variable Users


API for custom analysis
~~~~~~~~~~~~~~~~~~~~~~~

It is possible to get more details then the summary text in the above example.
The pass timings are stored in a
:class:`numba.misc.llvm_pass_timings.PassTimingsCollection`, which contains
methods for accessing individual record for each pass.

.. autoclass:: numba.misc.llvm_pass_timings.PassTimingsCollection
    :members: get_total_time, list_longest_first, summary, __getitem__, __len__

.. autoclass:: numba.misc.llvm_pass_timings.ProcessedPassTimings
    :members: get_raw_data, get_total_time, list_records, list_top, summary

.. autoclass:: numba.misc.llvm_pass_timings.PassTimingRecord
