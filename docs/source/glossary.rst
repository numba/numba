********
Glossary
********

.. glossary::

    JIT function
        Shorthand for "a function compiled with Numba."

    loop-lifting
        See :term:`loop-jitting`.

    loop-jitting
        A feature of compilation in :term:`object mode` where a loop can be
        automatically extracted and compiled in :term:`nopython mode`.  This
        allows functions with operations unsupported in nopython mode (such as
        array creation) to see significant performance improvements if they
        contain loops with only nopython-supported operations.

    nopython mode
        A Numba compilation mode that generates code that does not access the
        Python C API.  This compilation mode produces the highest performance
        code, but requires that the native types of all values in the function
        can be inferred, and that no new objects are allocated.  Unless
        otherwise instructed, the ``@jit`` decorator will automatically fall
        back to :term:`object mode` if nopython mode cannot be used.

    object mode
        A Numba compilation mode that generates code that handles all values
        as Python objects and uses the Python C API to perform all operations
        on those objects.  Code compiled in object mode will often run
        no faster than Python interpreted code, unless the Numba compiler can
        take advantage of :term:`loop-jitting`.

    type inference
        The process by which Numba determines the specialized types of all 
        values within a function being compiled.  Type inference can fail
        if arguments or globals have Python types unknown to Numba, or if
        functions are used that are not recognized by Numba.  Sucessful
        type inference is a prerequisite for compilation in
        :term:`nopython mode`.

    ufunc
        A NumPy `universal function <http://docs.scipy.org/doc/numpy/reference/ufuncs.html>`_.
        Numba can create new compiled ufuncs with the `@vectorize` decorator.
