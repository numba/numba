import numpy as np
from numba import *
from numbapro.vectorize.minivectorize import MiniVectorize as Vectorize
from time import time

def vector_add(a, b):
    return a + b

pv = Vectorize(vector_add)
pv.add(ret_type=int32, arg_types=[int32, int32])
pv.add(ret_type=f, arg_types=[f, f])
mini_ufunc = pv.build_ufunc()

test_dtypes = np.float32, np.int32

def test_nd():
    def test(dtype, order, nd, size=10, stride=1):
        # shape = tuple(range(size, size + nd))
        shape = (size,) * nd
        data = np.random.random(shape).astype(dtype)
        data[data != data] = 2.4
        data[data == float('inf')] = 3.8
        data[data == float('-inf')] = -3.8
        data2 = data.T.copy(order=order)

        idx = (slice(None, None, stride),) * nd
        data = data[idx]
        data2 = data2[idx]

        result = data + data2
        our_result = mini_ufunc(data, data2)
        assert np.allclose(result, our_result), (dtype, order)

    # test(np.float32, 'C', 2)
    for stride in (1, 2):
        for nd in range(1, 8):
            for dtype in test_dtypes:
                for order in ('C', 'F'):
                    test(dtype, order, nd, stride=stride)

def test_ufunc_attrib():
    assert mini_ufunc.reduce(np.arange(10, dtype=np.int32)) == 45

if __name__ == '__main__':
    test_nd()
    test_ufunc_attrib()
