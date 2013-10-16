from .support import testcase, main, assertTrue
import numpy as np
from numbapro import cuda, float32, int16


@cuda.jit(argtypes=[float32[:,:], int16, int16])
def div(grid, l_x, l_y):
    for x in range(l_x):
        for y in range(l_y):
            grid[x,y] /= 2.0

@testcase
def test_inplace_div():
    x = np.ones((2,2), dtype=np.float32)
    grid = cuda.to_device(x)
    div(grid, 2, 2)
    y = grid.copy_to_host()
    assertTrue(np.all(y == 0.5))

if __name__=='__main__':
    main()
