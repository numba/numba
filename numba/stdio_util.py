#! /usr/bin/env python
# ______________________________________________________________________

import ctypes
import ctypes.util

from numba import *

# ______________________________________________________________________

c_void_pp = ctypes.POINTER(ctypes.c_void_p)

def get_libc ():
    return ctypes.CDLL(ctypes.util.find_library('c'))

def get_stdio_streams ():
    '''
    Returns file pointers (FILE *) as Python integers for the C stdio
    stdin, stdout, and stderr streams.
    '''
    ret_val = None
    if hasattr(ctypes.pythonapi, 'stdin'):
        # Linux
        _stdio_files = (ctypes.c_void_p.in_dll(ctypes.pythonapi, sym)
                        for sym in ('stdin', 'stdout', 'stderr'))
        ret_val = tuple(c_void_pp(file_p)[0] for file_p in _stdio_files)
    elif hasattr(ctypes.pythonapi, '__stdinp'):
        # OSX
        _stdio_files = (ctypes.c_void_p.in_dll(ctypes.pythonapi, sym)
                        for sym in ('__stdinp', '__stdoutp', '__stderrp'))
        ret_val = tuple(c_void_pp(file_p)[0] for file_p in _stdio_files)
    else:
        libc = get_libc()
        if hasattr(libc, '__getreent'):
            # Cygwin
            ret_val = tuple(ctypes.cast(libc.__getreent(), c_void_pp)[1:4])
        else:
            raise NotImplementedError("Unsupported platform, don't know how to "
                                      "find pointers to stdio streams!")
    return ret_val

def get_stream_as_node(fp):
    return nodes.CoercionNode(nodes.ConstNode(fp, Py_ssize_t),
                              void.pointer())

# ______________________________________________________________________

def main ():
    _, stdout, _ = get_stdio_streams()
    PyObject_Print = ctypes.pythonapi.PyObject_Print
    PyObject_Print.restype = ctypes.c_int
    PyObject_Print.argtypes = ctypes.py_object, ctypes.c_void_p, ctypes.c_int
    PyObject_Print(get_stdio_streams, stdout, 1)
    PyObject_Print('\n\n', stdout, 1)

# ______________________________________________________________________

if __name__ == "__main__":
    main()

# ______________________________________________________________________
# End of stdio_util.py
