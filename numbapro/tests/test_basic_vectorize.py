import numpy as np
from numba import *
f, d = f4, f8
from numbapro.vectorize.basic import BasicVectorize
from time import time

def vector_add(a, b):
    return a + b

def main():
    # build basic native code ufunc
    bv = BasicVectorize(vector_add)
    bv.add(restype=int32, argtypes=[int32, int32])
    bv.add(restype=uint32, argtypes=[uint32, uint32])
    bv.add(restype=f, argtypes=[f, f])
    bv.add(restype=d, argtypes=[d, d])
    basic_ufunc = bv.build_ufunc()

    # build python ufunc
    np_ufunc = np.add

    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.linspace(0., 10000., 100000).astype(ty)

        ts = time()
        result = basic_ufunc(data, data)
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
            assert expect == got

    test(np.double)
    test(np.float32)
    test(np.int32)
    test(np.uint32)


    print('All good')

if __name__ == '__main__':
    main()

