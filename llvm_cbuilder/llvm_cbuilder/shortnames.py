from llvm.core import Type

void = Type.void()
char = Type.int(8)
short = Type.int(16)
int = Type.int(32)
int16 = short
int32 = int
int64 = Type.int(64)

float = Type.float()
double = Type.double()

# pointers

pointer = Type.pointer

void_p = pointer(char)

# platform dependent

def _determine_pointer_size():
    from ctypes import sizeof, c_void_p
    return sizeof(c_void_p) * 8

pointer_size = _determine_pointer_size()

intp = {32: int32, 64: int64}[pointer_size]

