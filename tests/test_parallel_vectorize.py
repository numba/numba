import numpy as np
from numba import *
from numbapro.parallel_vectorize import ParallelVectorize
from time import time

def vector_add(a, b):
    return a + b

def main():
    # build parallel native code ufunc
    pv = ParallelVectorize(vector_add)
    pv.add(ret_type=d, arg_types=[d, d])
    pv.add(ret_type=f, arg_types=[f, f])
    pv.add(ret_type=int32, arg_types=[int32, int32])
    para_ufunc = pv.build_ufunc()

    # build python ufunc
    np_ufunc = np.vectorize(vector_add)

    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.linspace(0., 10000., 100000).astype(ty)

        ts = time()
        result = para_ufunc(data, data)
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


    print('All good')

if __name__ == '__main__':
    main()
