import os.path
import numpy as np
from .support import testcase, main, assertTrue
from numbapro.cudadrv.driver import Linker, require_context
from numbapro import cuda


@testcase
@require_context
def test_linker_basic():
    '''Simply go through the constructor and destructor
    '''
    linker = Linker()
    del linker

@testcase
@require_context
def test_linking():
    global bar  # must be a global; other it is recognized as a freevar
    bar = cuda.declare_device('bar', 'int32(int32)')

    link = os.path.join(os.path.dirname(__file__), 'data', 'jitlink.o')
    print 'link to:', link

    if not os.path.isfile(link):
        print 'test skipped due to missing file'
        return

    @cuda.jit('void(int32[:], int32[:])', link=[link])
    def foo(x, y):
        i = cuda.grid(1)
        x[i] += bar(y[i])

    A = np.array([123])
    B = np.array([321])

    foo(A, B)

    assertTrue(A[0] == 123 + 2 * 321)

if __name__ == '__main__':
    main()
