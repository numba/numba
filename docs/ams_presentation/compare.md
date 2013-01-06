Numba, Cython and PyPy
----------------------

<!---
.. image:: cythonlogo.png
    :width: 50px
    :scale: 10 %
    :alt: alternate text
    :align: right
-->

* Ahead of time
* Compiles to C/C\+\+
    * build step
* Explicit types \+ type inference
    * Python semantics
        * quick fallback to objects

PyPy
----
* Python implementation
* Runtime tracing JIT


* Numba

    * Explicit runtime compilation
    * Type inferred
        - variable reuse
        - stronger typing
    * NumPy and Blaze aware
        - Focus on numerical and scientific computing

