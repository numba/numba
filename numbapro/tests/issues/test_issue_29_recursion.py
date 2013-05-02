import numpy as np
from numbapro import jit, cuda

if cuda.is_available:

    @jit('float32(float32)', target='gpu', device=True)
    def fact(x):
        if x <= 0:
            return 1
        else:
            return x * fact(x-1)


    @jit('void(float32[:])', target='gpu')
    def kfact(x):
        x[0] = fact(x[0])

    def test():
        a = np.array([4])
        kfact(a)

        print a

    test()
