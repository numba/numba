# Why Numba?

Python
------

* rapid iteration and development
* powerful libraries
* but, slow!

Numba
-----

* provides **JIT** and **array-oriented programming** in CPython
* works with existing CPython extensions
* Goal: Seamless integration with CPython at the core
    * integrate with scientific software stack
        * NumPy/SciPy/Blaze
    * C, C++, Fortran, Cython, CFFI, Julia?

# Array-Oriented Programming

- High level tools for domains experts to exploit modern hardware:
    - multicore CPU
    - manycore GPU

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

