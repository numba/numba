import sys
from numba import *

@jit
class Base(object):

    @void(double)
    def __init__(self, myfloat):
        self.value = myfloat

    @double()
    def getvalue(self):
        "Return value"
        return self.value

@jit
class Derived1(Base):

    @void(double)
    def __init__(self, value):
        self.value = value
        self.value2 = double(2.0)

def test_sizeof_extra_attr():
    base = Base(10.0)
    derived = Derived1(10.0)
    base_size = sys.getsizeof(base)
    derived_size = sys.getsizeof(derived)
    assert base_size + 8 == derived_size, (base_size, derived_size)

@jit
class Derived2(Base):

    @double()
    def getvalue(self):
        return self.value

def test_sizeof_extra_method():
    base_size = sys.getsizeof(Base(10.0))
    derived_size = sys.getsizeof(Derived2(10.0))
    assert base_size == derived_size, (base_size, derived_size)


if __name__ == '__main__':
    test_sizeof_extra_attr()
    test_sizeof_extra_method()