from .support import testcase, main, assertTrue
from numbapro import cuda, jit, float32, int32
import numpy as np


def foo(inp, out):
    for i in range(out.shape[0]):
        out[i] = inp[i]


def copy(inp, out):
    i = cuda.grid(1)
    cufoo(inp[i, :], out[i, :])


@testcase
def test_slice_as_arg():
    global cufoo
    cufoo = cuda.jit("void(int32[:], int32[:])", device=True)(foo)
    cucopy = cuda.jit("void(int32[:,:], int32[:,:])")(copy)

    inp = np.arange(100, dtype=np.int32).reshape(10, 10)
    out = np.zeros_like(inp)

    cucopy[1, 10](inp, out)


if __name__ == '__main__':
    main()
