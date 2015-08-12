from __future__ import print_function, absolute_import, division

from ctypes import *
import sys


is_windows = sys.platform.startswith('win32')

if is_windows:
    libc = cdll.msvcrt
else:
    libc = CDLL(None)

# A typed libc function (cdecl under Windows)

c_sin = libc.sin
c_sin.argtypes = [c_double]
c_sin.restype = c_double

def use_c_sin(x):
    return c_sin(x)

c_cos = libc.cos
c_cos.argtypes = [c_double]
c_cos.restype = c_double

def use_two_funcs(x):
    return c_sin(x) - c_cos(x)

# An untyped libc function

c_untyped = libc.exp

def use_c_untyped(x):
    return c_untyped(x)

# A libc function wrapped in a CFUNCTYPE

ctype_wrapping = CFUNCTYPE(c_double, c_double)(use_c_sin)

def use_ctype_wrapping(x):
    return ctype_wrapping(x)

# A Python API function

savethread = pythonapi.PyEval_SaveThread
savethread.argtypes = []
savethread.restype = c_void_p

restorethread = pythonapi.PyEval_RestoreThread
restorethread.argtypes = [c_void_p]
restorethread.restype = None

if is_windows:
    # A function with the stdcall calling convention
    c_sleep = windll.kernel32.Sleep
    c_sleep.argtypes = [c_uint]
    c_sleep.restype = None

    def use_c_sleep(x):
        c_sleep(x)


def use_c_pointer(x):
    """
    Running in Python will cause a segfault.
    """
    threadstate = savethread()
    x += 1
    restorethread(threadstate)
    return x


def use_func_pointer(fa, fb, x):
    if x > 0:
        return fa(x)
    else:
        return fb(x)


mydct = {'what': 1232121}

def call_me_maybe(arr):
    return mydct[arr[0].decode('ascii')]

# Create a callback into the python interpreter
py_call_back = CFUNCTYPE(c_int, py_object)(call_me_maybe)


def take_array_ptr(ptr):
    return ptr

c_take_array_ptr = CFUNCTYPE(c_void_p, c_void_p)(take_array_ptr)
