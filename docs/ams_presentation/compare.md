# Numba, Cython and PyPy

+---------------+---------------------------+---------------------------+-----------------------+
|               | Numba                     | Cython                    | PyPy                  |
+===============+===========================+===========================+=======================+
|               | - Runtime                 | - Ahead of time           | - Runtime tracing \   |
| **Compiling \ |     - Static or dynamic   |     - build step          |   JIT                 |
|   Strategy**  | - Ahead of time           |                           |                       |
|               |                           |                           |                       |
+---------------+---------------------------+---------------------------+-----------------------+
| **IR**        | - LLVM                    | - C/C++                   | - PyPy JIT            |
|               |                           |                           |                       |
+---------------+---------------------------+---------------------------+-----------------------+
| **Typing**    | - Type inferred           | - Explicit types & \      | - Full Python         |
|               | - Single type at each \   |   type inference          |   compatability       |
|               |   control flow point (\   |     - Quick fallback to \ |                       |
|               |   like RPython)           |                           |                       |
|               | - Variable reuse          |        objects            |                       |
|               | - Python semantics for    | - Python semantics for    |                       |
|               |   objects                 |   objects                 |                       |
+---------------+---------------------------+---------------------------+-----------------------+

Numba has a focus on numerical and scientific computing

