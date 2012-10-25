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

    @staticmethod
    @double(double, double)
    def static4(value1, value2):
        return value1 * value2

    @classmethod
    @double()
    def class1(cls):
        return 10.0

    @double(double)
    @classmethod
    def class2(cls, value):
        return value * 2

    @double(double, double)
    @classmethod
    def class3(cls, value1, value2):
        return value1 * value2

    @classmethod
    @double(double, double)
    def class4(cls, value1, value2):
        return value1 * value2

@autojit
class Derived(Base):
    pass

def test_staticmethods():
    def test(obj):
        assert obj.static1() == 10.0
        assert obj.static2(10.0) == 20.0
        assert obj.static3(5.0, 6.0) == 30.0
        assert obj.static4(5.0, 6.0) == 30.0

    test(Base(2.0))
    test(Derived(2.0))
    test(Base)
    test(Derived)

def test_classmethods():
    def test(obj):
        assert obj.class1() == 10.0
        assert obj.class2(10.0) == 20.0
        assert obj.class3(5.0, 6.0) == 30.0
        assert obj.class4(5.0, 6.0) == 30.0

    test(Base(2.0))
    test(Derived(2.0))
    test(Base)
    test(Derived)


if __name__ == '__main__':
    test_staticmethods()
    test_classmethods()