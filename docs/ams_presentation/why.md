# Why Python?

- Rapid development cycle
- Powerful libraries
- Allows interfacing with native code
    - Excellent for glue
- ... but, slow!
    - especially for computation-heavy code
        - numerical algorithms

# Why Numba?

- Provides **JIT** for **array-oriented programming** in CPython
    - Numerical loops
    - High level tools for domains experts to exploit modern hardware
        - multicore CPU
        - manycore GPU
    - Low-level C-like code in pure Python
        - pointers, structs, callbacks
- Works with existing CPython extensions
- Goal: Integration with scientific software stack
    - NumPy/SciPy/Blaze
        - Currently integrates with part of NumPy
            - indexing and slicing
            - array expressions
            - math

    - C, C++, Fortran, Cython, CFFI, Julia?

- Easily take advantage of parallelism and accelerators

<!--- Add graphic for array-oriented programming -->


# Software Stack

-![](software_stack.png)

# @jit, @autojit

- Instead of JIT-ing all Python code, we target the hotspot
- Use decorators to mark functions or classes for *just-in-time* compilation

Static runtime compilation:

```python
@jit(double(double[:, :]))
def func(array):
    ...
```

Dynamic just-in-time specialization:

```python
@autojit
def func(array):
    ...
```

