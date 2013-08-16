from .support import testcase, main
from ..compiler import compile
from ..types import uint32, float32, arraytype, void
import numpy as np

def diagproduct(c, a, b, data):
  startX = data[0]
  startY = data[0]
  gridX = data[1]
  gridY = data[1]
  #height = c.shape[0] # unsed
  width  = c.shape[1]

  y = 0
  for x in range(startX, width, gridX):
    for y in range(startY, width, gridY):
        c[y, x] = x

@testcase
def test_for_precondition():
    '''A sign-extension bug is causing the for loop to not run.
    '''
    N = 8
    A = np.arange(N * N, dtype=np.float32).reshape(N, N)
    B = np.arange(N, dtype=np.float32)
    C = np.zeros_like(A)
    C.fill(-1)
    data = np.array([0, 1], dtype=np.uint32)

    func = compile(diagproduct, void, [arraytype(float32, 2, 'C'),
                                    arraytype(float32, 2, 'C'),
                                    arraytype(float32, 1, 'C'),
                                    arraytype(uint32, 1, 'C')])

    func(C, A, B, data)

    assert np.all(C != -1)

if __name__ == '__main__':
    main()
