
.. _c:

******************
Interfacing with C
******************

Ctypes and CFFI
===============
Numba supports jitting ctypes and CFFI function calls. Numba will automatically
figure out the signatures of the functions and data. Below is a gibbs sampling
code that accesses ctypes (or CFFI) functions defined in another module (
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
Users can access and define new or external stucts, deal with pointers, and
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


