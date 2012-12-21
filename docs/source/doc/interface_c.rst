
.. _c:

***************************************
Primitive values and Interfacing with C
***************************************

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
