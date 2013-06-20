
.. _c:

******************
Interfacing with C
******************

Ctypes and CFFI
===============
Numba supports jitting ctypes and CFFI function calls. Numba will automatically
figure out the signatures of the functions and data. Below is a Gibbs sampling
implementation that accesses ctypes (or CFFI) functions defined in another
module (
``rk_seed``,  ``rk_gamma`` and ``rk_normal``), and that passes in a pointer to a struct
also allocated with ctypes (``state_p``)::

    def gibbs(N, thin):
        rng.rk_seed(0, rng.state_p)

        x = 0
        y = 0
        samples = np.empty((N,2))
        for i in range(N):
            for j in range(thin):
                x = rng.rk_gamma(rng.state_p, 3.0, 1.0/(y**2+4))
                y = rng.rk_normal(rng.state_p, 1.0/(x+1), 1.0/math.sqrt(2+2*x))

            samples[i, 0] = x
            samples[i, 1] = y

        return samples

.. NOTE:: Passing in ctypes or CFFI libraries to autojit functions does not yet work.
          However, passing in individual functions does.

Writing Low-level Code
======================

Numba allows users to deal with high-level as well as low-level C-like code.
Users can access and define new or external structs, deal with pointers, and
call C functions natively.

.. NOTE:: Type declarations are covered in in :ref:`types`.

Structs
-------
Structs can be declared on the stack, passed in as arguments,
returned from external or Numba
functions, or originate from record arrays. Structs currently have
copy-by-value semantics, but this is likely to change.

Struct fields can accessed as follows:

    * ``struct.attr``
    * ``struct['attr']``
    * ``struct[field_index]``

.. An example can be found here: :ref:`structexample`.
An example is shown below:

.. literalinclude:: /../../examples/structures.py

Pointers
--------
Pointers in Numba can be used in a similar way to C. They can be cast,
indexed and operated on with pointer arithmetic. Currently it is however
not possible to obtain the address of an lvalue.

.. An example can be found here: :ref:`pointerexample`.
An example is shown below:

.. literalinclude:: /../../examples/pointers.py

.. _intrinsics:

Using Intrinsics
================

Numba allows users to declare and use LLVM intrinsics and instructions directly.
This allows one to implement very low-level features directly in Python, while
maintaining compatibility with Python code not compiled with numba itself.

.. function:: declare_intrinsic(func_signature, intrinsic_name)

   This declares an LLVM intrinsic function with the given name.
   E.g.

::

    # declare i64 @llvm.readcyclecounter()
    intrin = numba.declare_intrinsic(int64(), "llvm.readcyclecounter")
    print intrin()

.. NOTE:: Intrinsics are not yet implemented. Only the instructions below are.

.. function:: declare_instruction(func_signature, intrinsic_name)

   This declares an LLVM instruction named ``name`` as a function.
   E.g. we can use the ``srem`` instruction [#]_ to calculate the remainder
   of a signed integer, in an equivalent manner to how modulo works
   in C (see [#]_ and [#]_ for how this differs from Python).

::


    >>> rem = nb.declare_instruction(int32(int32, int32), 'srem')
    >>>
    >>> @jit(int32(int32, int32))
    ... def py_modulo(a, n):
    ...     r = rem(a, n)
    ...     if r != 0 and (r ^ n) < 0:
    ...         r += n
    ...     return r

    >>> # Instructions and intrinsics works directly in Python
    >>> print rem(5, 2), rem(5, -2), rem(-5, 2), rem(-5, -2)
    1 1 -1 -1

    >>> # ... and are jitted in Numba functions
    >>> print py_modulo(5, 2), py_modulo(5, -2), py_modulo(-5, 2), py_modulo(-5, -2)
    1 -1 1 -1

    >>> print py_modulo.lfunc
    define i32 @__numba_specialized_6___main___2E_py_modulo(i32 %a, i32 %n) nounwind readnone {
    entry:
      %0 = srem i32 %a, %n
      %1 = icmp ne i32 %0, 0
      %2 = xor i32 %0, %n
      %3 = icmp slt i32 %2, 0
      %or.cond = and i1 %1, %3
      %4 = select i1 %or.cond, i32 %n, i32 0
      %. = add i32 %4, %0
      ret i32 %.
    }

As you can see the instructions can be used in Numba, where the instruction
is inserted directly in the instruction stream, or pure-Python, where arguments
are converted to native values, the operation executed, and the result returned
as a Python object. We can verify that our modulo function works in pure Python::

    >>> print py_modulo.py_func(5,  2), py_modulo.py_func( 5, -2), \
    ...       py_modulo.py_func(-5, 2), py_modulo.py_func(-5, -2)
    1 -1 1 -1

.. NOTE:: Numba does not validate the signatures or validity of instructions
          and intrinsics. This is the responsibility of the user. Fortunately,
          the validity can be quickly verified :)

Using Numba Functions in External Code
======================================

Users can take the address of a numba compiled function using ``numba.addressof``:

.. function:: addressof(jit_func, propagate=True)

    Take the address of ``jit_func`` as a ctypes function. ``propagate`` indicates
    whether uncaught exceptions propogate or are written to stderr.

::

    @jit(int32(int32, int32))
    def mul(a, b):
        return a * b

    cmul = numba.addressof(mul)
    print cmul(5, 2)

    # Get the address as an Python int
    addr_int = ctypes.cast(cmul, ctypes.c_void_p).value
    print hex(addr_int)

Callers can currently check for exceptions (where appropriate) using
``PyErr_Occurred()`` (which requires the GIL).

``nopython`` functions
which do not directly or indirectly call functions requiring the GIL or
use the ``with python`` construct can be called without the GIL. Numba
does not check whether this is valid, nor does it currently acquire the GIL.

Currently supported bad values for return types:

=================   =============
Return Type         Bad Value
=================   =============
``object_``         ``NULL``
``floating``        ``NaN``
=================   =============

These can be checked as follows:

.. code-block:: c

    float ret = my_numba_func(...);
    if (ret != ret && PyErr_Occurred()) {
        // Handle error
    }

The error indicator for integer values is currently undecided, since
constants in LLVM bitcode are printed in decimal form, but hex codes
can be a better choice for other scenarios.

References
==========
.. [#] http://llvm.org/docs/LangRef.html#srem-instruction
.. [#] http://en.wikipedia.org/wiki/Modulo_operation
.. [#] http://wiki.cython.org/enhancements/division

