.. _developer-llvm-timings:

====================
Notes on timing LLVM
====================


Getting LLVM Pass Timings
-------------------------

The dispatcher stores LLVM pass timings in the dispatcher object metadata under the
``llvm_pass_timings`` key. The timings information contains details on how much time
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
   :dedent: 12
   :linenos:

Example output:

.. code-block:: text

    Printing pass timings for JITCodeLibrary('DocsLLVMPassTimings.test_pass_timings.<locals>.foo')
    Total time: 0.0360
    == #0 Function passes '_ZN5numba5tests12doc_examples22test_llvm_pass_timings19DocsLLVMPassTimings17test_pass_timings12$3clocals$3e7foo$241Ex'
     Percent: 5.0%
     Total 0.0018s
     Top timings:
       0.0015s ( 81.5%) SROA #3
       0.0002s (  9.3%) Early CSE #2
       0.0001s (  4.3%) Simplify the CFG #9
       0.0000s (  1.4%) Prune NRT refops #4
       0.0000s (  1.0%) Post-Dominator Tree Construction #5
    == #1 Function passes '_ZN7cpython5numba5tests12doc_examples22test_llvm_pass_timings19DocsLLVMPassTimings17test_pass_timings12$3clocals$3e7foo$241Ex'
     Percent: 0.8%
     Total 0.0003s
     Top timings:
       0.0001s ( 32.1%) Simplify the CFG #10
       0.0001s ( 24.7%) Lower 'expect' Intrinsics #2
       0.0000s ( 15.2%) SROA #4
       0.0000s (  8.8%) Prune NRT refops #5
       0.0000s (  5.5%) Post-Dominator Tree Construction #6
    == #2 Function passes 'cfunc._ZN5numba5tests12doc_examples22test_llvm_pass_timings19DocsLLVMPassTimings17test_pass_timings12$3clocals$3e7foo$241Ex'
     Percent: 0.3%
     Total 0.0001s
     Top timings:
       0.0000s ( 27.2%) Lower 'expect' Intrinsics #3
       0.0000s ( 26.7%) Simplify the CFG #11
       0.0000s ( 13.2%) Prune NRT refops #6
       0.0000s (  7.7%) Post-Dominator Tree Construction #7
       0.0000s (  7.0%) Dominator Tree Construction #29
    == #3 Module passes (cheap)
     Percent: 3.6%
     Total 0.0013s
     Top timings:
       0.0007s ( 54.3%) Combine redundant instructions
       0.0001s (  5.1%) Function Integration/Inlining
       0.0001s (  4.7%) Natural Loop Information
       0.0001s (  4.6%) Prune NRT refops #2
       0.0000s (  3.3%) Dominator Tree Construction #3
    == #4 Module passes (full)
     Percent: 43.3%
     Total 0.0156s
     Top timings:
       0.0030s ( 19.2%) Combine redundant instructions #9
       0.0021s ( 13.3%) Combine redundant instructions #7
       0.0010s (  6.6%) Induction Variable Simplification
       0.0007s (  4.8%) Unroll loops #2
       0.0007s (  4.8%) Loop Vectorization
    == #5 Finalize object
     Percent: 46.9%
     Total 0.0169s
     Top timings:
       0.0059s ( 35.2%) X86 DAG->DAG Instruction Selection #2
       0.0019s ( 11.1%) Greedy Register Allocator #2
       0.0014s (  8.0%) Machine Instruction Scheduler #2
       0.0009s (  5.6%) Loop Strength Reduction
       0.0004s (  2.1%) Induction Variable Users


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
