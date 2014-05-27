========================
Code Generation Utilties
========================

.. contents::

Introduction
=============

Numba uses llvmpy for code generation.  The ``numba/cgutils.py`` file
provides utilities to simplify many common tasks.

Structures
============

The ``numba.cgutils.Structure`` class is used like the ``ctypes.Structure``
for manuipluting structures.  The ``_fields`` class
attribute declares a list of (name, type) pairs for each field of the
structure.

For example, The complex number structures are defined as::

    from numba import types, cgutils

    class Complex64(cgutils.Structure):
        _fields = [('real', types.float32),
                   ('imag', types.float32)]


    class Complex128(cgutils.Structure):
        _fields = [('real', types.float64),
                   ('imag', types.float64)]


The constructor of ``Structure`` requires the codegen context and
llvm builder.  Optionally, user can provide the ``value`` keyword argument
to initialize the structure with a LLVM value of compatible type.  Structures
are allocated using ``llvm.Builder.alloca``.  (The ``alloca`` operation is
automatically placed as the first block of the current function.)
User can provide the ``ref`` keyword argument to specify a
custom storage space.  It should contain a LLVM pointer of compatible type.
To get the pointer to the allocated space,
use ``Structure._getpointer``.  To get a copy of the structure as a
value, use ``Structure._getvalue``.  The two functions returns a LLVM value
(instance of ``llvm.core.Value``).

Each field is a load/store property of the ``Structure`` instance.  Accessing
the ``real`` field of ``Complex64``, for example, looks like::


    c = Complex64(context, builder)
    c.real = llvm.core.Constant.real(lc.Type.float32, 1.23)
    two_times_real = builder.add(c.real, c.real)

