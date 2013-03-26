from numba import *

@jit # Test compiling this class
class Test(object):
    @void(double,double)
    def __init__(self, a, b):
        self.a = a
        self.b = b
        for i in range(self.a, self.b):
            pass
        self.i = i

if __name__ == "__main__":
    pass
