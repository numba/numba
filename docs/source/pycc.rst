Static Compilation (pycc)
=========================
``pycc`` allows users to compile Numba functions into a shared library.
The user writes the functions, exports them and the compiler will import
the module, collect the exported functions and compile them to a shared
library. Below is an example::

    from numba import *

    def mult(a, b):
        return a * b

    export('mult f8(f8, f8)')(mult)
    exportmany(['multf f4(f4, f4)', 'multi i4(i4, i4)'])(mult)
    export('multc c16(c16, c16)')(mult)

This defines a trivial function and exports four specializations under
different names. The code can be compiled as follows::

    pycc thefile.py

Which will create a pure shared library for your platform which can be
linked against any other program.  This is **not** a Python extension.
You would have to use ctypes to load the code that is created.
Multiple files may be given to compile them simultaneously into a
shared library. Options exist to compile to native object files
instead of a shared library, to emit LLVM code or to generate a C
header file with function prototypes. For more information on the
available command line options, see ``pycc -h``.
