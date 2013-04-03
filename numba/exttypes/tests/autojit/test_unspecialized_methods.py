from numba import *
from numba.testing.test_support import parametrize, main

from numba.exttypes.tests import test_extension_methods

Base1 = test_extension_methods.make_base(autojit)

@autojit
class Base2(object):

    def __init__(self, myfloat):
        self.value = myfloat

    def getvalue(self):
        return self.value

    # @staticmethod
    # def static1():
    #     return 10.0
    #
    # @staticmethod
    # def static2(value):
    #     return value * 2
    #
    # @staticmethod
    # def static3(value1, value2):
    #     return value1 * value2
    #
    # @staticmethod
    # @double(double, double)
    # def static4(value1, value2):
    #     return value1 * value2
    #
    # @classmethod
    # @double()
    # def class1(cls):
    #     return 10.0
    #
    # @double(double)
    # @classmethod
    # def class2(cls, value):
    #     return value * 2
    #
    # @double(double, double)
    # @classmethod
    # def class3(cls, value1, value2):
    #     return value1 * value2
    #
    # @classmethod
    # @double(double, double)
    # def class4(cls, value1, value2):
    #     return value1 * value2

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

obj1 = Base1(10.0)
specialized_class1 = Base1[{'value': double}]
obj2 = Base1(11)

@parametrize(Base1, specialized_class1, obj1, obj2)
def test_staticmethods(obj):
    assert obj.static1() == 10.0
    assert obj.static2(10.0) == 20.0
    assert obj.static3(5.0, 6.0) == 30.0
    assert obj.static4(5.0, 6.0) == 30.0

@parametrize(Base1, specialized_class1, obj1, obj2)
def test_classmethods(obj):
    assert obj.class1() == 10.0
    assert obj.class2(10.0) == 20.0
    assert obj.class3(5.0, 6.0) == 30.0
    assert obj.class4(5.0, 6.0) == 30.0

@parametrize(obj1)
def test_specialized_unbound(obj):
    assert type(obj) is specialized_class1
    assert specialized_class1.getvalue(obj) == 10.0

@parametrize(obj2)
def test_specialized_unbound2(obj):
    assert issubclass(type(obj), Base1)
    assert type(obj).getvalue(obj) == 11


if __name__ == '__main__':
    main()
