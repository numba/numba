import ctypes

from numba import *

@autojit(backend='ast')
def call_ctypes(func):
    return func("Hello %s\n", "World!")

def test_ctypes_calls():
    libc = ctypes.CDLL(ctypes.util.find_library('c'))
    printf = libc.printf
    printf.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    #printf("hello %d\n", 10)
    print call_ctypes(printf)

if __name__ == "__main__":
    test_ctypes_calls()