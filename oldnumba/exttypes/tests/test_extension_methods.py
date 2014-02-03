from numba import *
from numba.testing.test_support import parametrize, main

def make_base(compiler):
    @compiler
    class Base(object):

        @void(double)
        def __init__(self, myfloat):
            self.value = myfloat

        @double()
        def getvalue(self):
            return self.value

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

    return Base

def make_derived(compiler):
    Base = make_base(compiler)

    @compiler
    class Derived(Base):
        pass

    return Base, Derived

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

def run_staticmethods(obj):
    assert obj.static1() == 10.0
    assert obj.static2(10.0) == 20.0
    assert obj.static3(5.0, 6.0) == 30.0
    assert obj.static4(5.0, 6.0) == 30.0

def run_classmethods(obj):
    assert obj.class1() == 10.0
    assert obj.class2(10.0) == 20.0
    assert obj.class3(5.0, 6.0) == 30.0
    assert obj.class4(5.0, 6.0) == 30.0

# ______________________________________________________________________
# Parameterized tests

@parametrize(jit, autojit)
def test_base_staticmethods(compiler):
    Base = make_base(compiler)
    run_staticmethods(Base(2.0))
    run_staticmethods(Base)

@parametrize(jit)
def test_derived_staticmethods(compiler):
    Base, Derived = make_derived(compiler)
    run_staticmethods(Derived(2.0))
    run_staticmethods(Derived)

@parametrize(jit, autojit)
def test_base_classmethods(compiler):
    Base = make_base(compiler)
    run_classmethods(Base(2.0))
    run_classmethods(Base)

@parametrize(jit)
def test_derived_classmethods(compiler):
    Base, Derived = make_derived(compiler)
    run_classmethods(Derived(2.0))
    run_classmethods(Derived)

if __name__ == '__main__':
    # Base = make_base(autojit)
    # obj = Base(2.0)
    # run_staticmethods(Base)
    main()
