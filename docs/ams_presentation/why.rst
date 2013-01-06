Why Numba?
------------

- Python
    - rapid iteration and development
    - powerful libraries
    - but, slow!

- Numba
    - provides **JIT** and **array-oriented programming** in CPython
    - works with existing CPython extensions

Array-Oriented Programming
---------------------------

- High level tools for domains experts to exploit modern hardware:
    - multicore CPU
    - manycore GPU
    
- Easily take advantage of parallelism and accelerators

.. Add graphic for array-oriented programming


Software Stack
---------------

.. image:: software_stack.png

@jit, @autojit
--------------

- Instead of JIT-ing all Python code, we target the hotspot.
- Use decorators to mark functions or classes for *just-in-time* compilation.





