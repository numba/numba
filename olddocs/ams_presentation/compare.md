# Compiling Strategy: Numba vs Cython vs PyPy

+---------------------------+---------------------------+-----------------------+
| Numba                     | Cython                    | PyPy                  |
+===========================+===========================+=======================+
| - Runtime                 | - Ahead of time           | - Runtime tracing     |
|     - Static or dynamic   |     - build step          |   JIT                 |
| - Ahead of time           |                           |                       |
|                           |                           |                       |
+---------------------------+---------------------------+-----------------------+

# Compiler IR: Numba vs Cython vs PyPy

+---------------------------+---------------------------+-----------------------+
| Numba                     | Cython                    | PyPy                  |
+===========================+===========================+=======================+
| - LLVM                    | - C/C++                   | - PyPy JIT            |
|                           |                           |                       |
+---------------------------+---------------------------+-----------------------+


# Typing: Numba vs Cython vs PyPy

+---------------------------+---------------------------+-----------------------+
| Numba                     | Cython                    | PyPy                  |
+===========================+===========================+=======================+
| - Type inferred           | - Explicit types &        | - Full Python         |
| - Single type at each     |   type inference          |   compatability       |
|   control flow point (    | - Quick fallback to       |                       |
|   like RPython)           |   objects                 |                       |
| - Variable reuse          |                           |                       |
| - Python semantics for    | - Python semantics for    |                       |
|   objects                 |   objects                 |                       |
+---------------------------+---------------------------+-----------------------+


