'''Example: sum each row using guvectorize

See Numpy documentation for detail about gufunc:
    http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
'''
import numpy as np
from numbapro import guvectorize, cuda
from .support import testcase, main

@testcase
def test_gufunc_scalar_output():
#    function type:
#        - has no void return type
#        - array argument is one dimenion fewer than the source array
#        - scalar output is passed as a 1-element array.
#
#    signature: (n)->()
#        - the function takes an array of n-element and output a scalar.

    @guvectorize(['void(int32[:], int32[:])'], '(n)->()', target='gpu')
    def sum_row(inp, out):
        tmp = 0.
        for i in range(inp.shape[0]):
            tmp += inp[i]
        out[0] = tmp

    # inp is (10000, 3)
    # out is (10000)
    # The outter (leftmost) dimension must match or numpy broadcasting is performed.
    # But, broadcasting on CUDA arrays is not supported.

    inp = np.arange(300, dtype=np.int32).reshape(100, 3)

    # invoke on CUDA with manually managed memory
    out = np.empty(100, dtype=inp.dtype)

    dev_inp = cuda.to_device(inp)             # alloc and copy input data
    dev_out = cuda.to_device(out, copy=False) # alloc only

    sum_row(dev_inp, out=dev_out)             # invoke the gufunc

    dev_out.copy_to_host(out)                 # retrieve the result

    # verify result
    for i in xrange(inp.shape[0]):
        assert out[i] == inp[i].sum()

if __name__ == '__main__':
    main()
