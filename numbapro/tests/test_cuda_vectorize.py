import numpy as np
from numba import *
from numbapro.vectorize import Vectorize
from time import time

def vector_add(a, b):
    return a + b

# build cuda code ufunc
pv = Vectorize(vector_add, target='gpu')
pv.add(restype=int32, argtypes=[int32, int32])
pv.add(restype=f4, argtypes=[f4, f4])
#pv.add(restype=d, argtypes=[d, d])
cuda_ufunc = pv.build_ufunc()

test_dtypes = np.float32, np.int32

def test_1d():
    # build python ufunc
    np_ufunc = np.add

    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.linspace(0., 10000., 500*501).astype(ty)

        ts = time()
        result = cuda_ufunc(data, data)
        tnumba = time() - ts

        ts = time()
        gold = np_ufunc(data, data)
        tnumpy = time() - ts

        print("Numpy time: %fs" % tnumpy)
        print("Numba time: %fs" % tnumba)

        if tnumba < tnumpy:
            print("Numba is FASTER by %fx" % (tnumpy/tnumba))
        else:
            print("Numba is SLOWER by %fx" % (tnumba/tnumpy))


        for expect, got in zip(gold, result):
            if got == 0:
                err = abs(expect - got)
            else:
                err = abs(expect - got)/float(got)
            if err > 1e-5:
                raise ValueError(expect, got, err)

    #test(np.double)
    test(np.float32)
    test(np.int32)

    print('All good')

def test_nd():
    def test(dtype, order, nd, size=10):
        data = np.random.random((size,) * nd).astype(dtype)
        data[data != data] = 2.4
        data[data == float('inf')] = 3.8
        data[data == float('-inf')] = -3.8
        data2 = np.array(data.T, order=order) #.copy(order=order)

        result = data + data2
        our_result = cuda_ufunc(data, data2)
        assert np.allclose(result, our_result), (dtype, order)

    for nd in range(1, 8):
        for dtype in test_dtypes:
            for order in ('C', 'F'):
                test(dtype, order, nd)

def test_ufunc_attrib():
    assert cuda_ufunc.reduce(np.arange(10, dtype=np.int32)) == 45

if __name__ == '__main__':
    test_ufunc_attrib()
    test_nd()
    test_1d()
