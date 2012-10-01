Static Compilation of Numba Functions
=====================================
``pycc`` allows users to compile Numba functions into a shared library.
The user writes the functions, exports them and the compiler will import
the module, collect the exported functions and compile them to a shared
library. Below is an example::

    from numba import *

    def mult(a, b):
        return a * b

    export(argtypes=[float64, float64], restype=float64)(mult, name='mult')
    export(argtypes=[float32, float32], restype=float32)(mult, name='multf')
    export(argtypes=[int32, int32], restype=int32)(mult, name='multi')
    export(argtypes=[complex128, complex128], restype=complex128)(mult, name='multc')

This defines a trivial function and exports four specializations under
different names. The code can be compiled as follows::

    pycc thefile.py

Which will create a shared library for your platform. Multiple files may be given
to compile them simulteneously into a shared library. Options exist to compile
to native object files instead of a shared library, to emit LLVM code or
to generate a C header file with function prototypes. For more information
on the available command line options, see ``pycc -h``.
