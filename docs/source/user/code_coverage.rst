===============================
Code Coverage for Compiled Code
===============================

Numba, a just-in-time compiler for Python, transforms Python code into machine 
code for optimized execution. This process, however, poses a challenge for 
traditional code coverage tools, as they typically operate within the Python 
interpreter and thus miss the lines of code compiled by Numba. To address this 
issue, Numba opts for a compile-time notification to coverage tools, rather than 
during execution, to minimize performance penalties. This approach helps prevent 
significant coverage gaps in projects utilizing Numba, without incurring 
substantial performance costs.

No additional effort is required to generate compile-time coverage data. By 
running a Numba application under the ``coverage`` tool 
(e.g. ``coverage run ...``), the compiler automatically 
detects the active coverage session and emits data accordingly. This mechanism 
ensures that coverage data is generated seamlessly, without the need for manual 
intervention.

The coverage data is emitted during the lowering phase, which involves the 
generation of LLVM-IR. This phase inherently excludes lines of code that are 
statically identified as dead code, ensuring that the coverage data accurately 
reflects the executable code paths.
