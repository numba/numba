import numpy as np
from numba import *
from numbapro.vectorize import Vectorize
from numbapro import cuda
from timing import time

def vector_add(a, b):
    return a + b

# build cuda code ufunc
pv = Vectorize(vector_add, target='gpu')
pv.add(restype=int32, argtypes=[int32, int32])
pv.add(restype=f4, argtypes=[f4, f4])
pv.add(restype=f8, argtypes=[f8, f8])
cuda_ufunc = pv.build_ufunc()

test_dtypes = np.float32, np.int32

def test_1d():
    # build python ufunc
    np_ufunc = np.add

    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.array(np.random.random(1e+6 + 1), dtype=ty)

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


        assert np.allclose(gold, result), (gold, result)

    test(np.double)
    test(np.float32)
    test(np.int32)


def test_1d_async():
    # build python ufunc
    np_ufunc = np.add

    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.array(np.random.random(1e+6 + 1), dtype=ty)

        ts = time()
        stream = cuda.stream()
        device_data = cuda.to_device(data, stream)
        dresult = cuda_ufunc(device_data, device_data, stream=stream)
        result = dresult.copy_to_host()
        stream.synchronize()
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


        assert np.allclose(gold, result), (gold, result)

    test(np.double)
    test(np.float32)
    test(np.int32)



def test_nd():
    def test(dtype, order, nd, size=10):
        data = np.random.random((size,) * nd).astype(dtype)
        data[data != data] = 2.4
        data[data == float('inf')] = 3.8
        data[data == float('-inf')] = -3.8
        data2 = np.array(data.T, order=order) #.copy(order=order)

        result = data + data2
        our_result = cuda_ufunc(data, data2)
        assert np.allclose(result, our_result), (dtype, order, result, our_result)

    for nd in range(1, 8):
        for dtype in test_dtypes:
            for order in ('C', 'F'):
                test(dtype, order, nd)

def test_ufunc_attrib():
    test_reduce(8)
    test_reduce(100)
    test_reduce(2**10 + 1)
    test_reduce2(8)
    test_reduce2(100)
    test_reduce2(2**10 + 1)

def test_reduce(n):
    x = np.arange(n, dtype=np.int32)
    gold = np.add.reduce(x)
    result = cuda_ufunc.reduce(x)
    assert result == gold, (result, gold)


def test_reduce2(n):
    x = np.arange(n, dtype=np.int32)
    gold = np.add.reduce(x)
    stream = cuda.stream()
    dx = cuda.to_device(x, stream)
    result = cuda_ufunc.reduce(x, stream=stream)
    assert result == gold, (result, gold)


if __name__ == '__main__':
    test_ufunc_attrib()
    test_nd()
    test_1d()
    test_1d_async()
