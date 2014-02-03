"""
Test class attributes.
"""

import numba
from numba import *
from numba.testing.test_support import parametrize, main

def make_base(compiler):
    @compiler
    class Base(object):

        value1 = double
        value2 = int_

        @void(int_, double)
        def __init__(self, value1, value2):
            self.value1 = value1
            self.value2 = value2

        @void(int_)
        def setvalue(self, value):
            self.value1 = value

        @double()
        def getvalue1(self):
            return self.value1

    return Base

def make_derived(compiler):
    Base = make_base(compiler)

    @compiler
    class Derived(Base):

        value3 = float_

        @void(int_)
        def setvalue(self, value):
            self.value3 = value

    return Base, Derived

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

@parametrize(jit, autojit)
def test_baseclass_attrs(compiler):
    Base = make_base(compiler)

    assert Base(10, 11.0).value1 == 10.0
    assert Base(10, 11.0).value2 == 11

    obj = Base(10, 11.0)
    obj.setvalue(12)
    assert obj.getvalue1() == 12.0

@parametrize(jit) #, autojit)
def test_derivedclass_attrs(compiler):
    Base, Derived = make_derived(compiler)

    obj = Derived(10, 11.0)
    obj.setvalue(9)
    assert obj.value3 == 9.0


if __name__ == '__main__':
    # test_derivedclass_attrs(autojit)
    main()
