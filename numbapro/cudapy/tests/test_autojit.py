import numpy as np
import math
import time
from numbapro import cuda
from .support import testcase, main

@testcase
def test_autojit():
    @cuda.autojit
    def what(a, b, c):
        pass

    what(np.empty(1), 1.0, 21)
    what(np.empty(1), 1.0, 21)
    what(np.empty(1), np.empty(1, dtype=np.int32), 21)
    what(np.empty(1), np.empty(1, dtype=np.int32), 21)
    what(np.empty(1), 1.0, 21)

    print what.definitions
    assert len(what.definitions) == 2

if __name__ == '__main__':
    main()

