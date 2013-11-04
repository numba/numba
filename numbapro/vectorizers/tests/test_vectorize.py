import numpy as np
from numba import int32, uint32, float32, float64
from numbapro import vectorize
from timeit import default_timer as time
from .support import testcase, main, assertTrue

def vector_add(a, b):
    return a + b

def template_vectorize(target):
    # build basic native code ufunc
    sig = [int32(int32, int32),
           uint32(uint32, uint32),
           float32(float32, float32),
           float64(float64, float64)]
    basic_ufunc = vectorize(sig, target=target)(vector_add)

    # build python ufunc
    np_ufunc = np.add

    # test it out
    def test(ty):
        print("Test %s" % ty)
        data = np.linspace(0., 100., 500).astype(ty)

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

        assertTrue(np.allclose(gold, result))

    test(np.double)
    test(np.float32)
    test(np.int32)
    test(np.uint32)

@testcase
def test_basic_vectorize():
    template_vectorize('cpu')

@testcase
def test_parallel_vectorize():
    template_vectorize('parallel')

@testcase
def test_stream_vectorize():
    template_vectorize('stream')

if __name__ == '__main__':
    main()
