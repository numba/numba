from numba import *
from numba import declare_intrinsic, declare_instruction

def test_intrinsics():
    intrin = declare_instruction(int32(int32, int32), 'srem')
    assert intrin(5, 3) == 2

if __name__ == "__main__":
    test_intrinsics()
