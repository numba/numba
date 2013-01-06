# Numba, Cython and PyPy

+---------------------------+-----------------------+---------------------------+
| Cython                    | PyPy                  | Numba                     |
+===========================+=======================+===========================+
| - Ahead of time           | - Runtime             | - Runtime                 |
|     - build step          |                       |     - Static or dynamic   |
|                           |                       | - Ahead of time           |
+---------------------------+-----------------------+---------------------------+
| - Compiles to             | - Tracing JIT         | - LLVM                    |
|   C/C++                   |                       |                           |
+---------------------------+-----------------------+---------------------------+
| - Explicit types &        | - Full Python         | - Type inference          |
|                           |                       |                           |
|                           |                       |                           |
|   type inference          |   compatability       |     - Stronger typing     |
|     - Python semantics    |                       |                           |
|     - Quick fallback to   |                       |                           |
|       objects             |                       |                           |
+---------------------------+-----------------------+---------------------------+


* Numba

    * Explicit runtime compilation
    * Type inferred
        - variable reuse
        - stronger typing
    * NumPy and Blaze aware
        - Focus on numerical and scientific computing


