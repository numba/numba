import sys
from numba import *

@autojit
class Base(object):

    @void(double)
    def __init__(self, myfloat):
        self.value = myfloat

    @double()
    def getvalue(self):
        return self.value
#
    @staticmethod
    @double()
    def static1():
        return 10.0

    @double(double)
    @staticmethod
    def static2(value):
        return value * 2

    @double(double, double)
    @staticmethod
    def static3(value1, value2):
        return value1 * value2

def test_staticmethods():
    obj = Base(2.0)
    assert obj.static1() == 10.0
    assert obj.static2(10.0) == 20.0
    assert obj.static3(5.0, 6.0) == 30.0


if __name__ == '__main__':
    test_staticmethods()