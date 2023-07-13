Libdevice functions
===================

All wrapped libdevice functions are listed in this section. All functions in
libdevice are wrapped, with the exception of ``__nv_nan`` and ``__nv_nanf``.
These functions return a representation of a quiet NaN, but the argument they
take (a pointer to an object specifying the representation) is undocumented, and
follows an unusual form compared to the rest of libdevice - it is not an output
like every other pointer argument. If a NaN is required, one can be obtained in
CUDA Python by other means, e.g. ``math.nan``.

Wrapped functions
-----------------

.. automodule:: numba.cuda.libdevice
   :members:
