from numba import *
from numba.testing.test_support import parametrize, main

from numba.exttypes.tests import test_extension_methods

Base = test_extension_methods.make_base(autojit)

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

obj = Base(10.0)
specialized_class = Base[{'value': double}]

obj2 = Base(11)

@parametrize(Base, specialized_class, obj, obj2)
def test_staticmethods(obj):
    assert obj.static1() == 10.0
    assert obj.static2(10.0) == 20.0
    assert obj.static3(5.0, 6.0) == 30.0
    assert obj.static4(5.0, 6.0) == 30.0

@parametrize(Base, specialized_class, obj, obj2)
def test_classmethods(obj):
    assert obj.class1() == 10.0
    assert obj.class2(10.0) == 20.0
    assert obj.class3(5.0, 6.0) == 30.0
    assert obj.class4(5.0, 6.0) == 30.0

@parametrize(obj)
def test_specialized_unbound(obj):
    assert type(obj) is specialized_class
    assert specialized_class.getvalue(obj) == 10.0

# @parametrize(obj2)
# def test_specialized_unbound2(obj):
#     assert issubclass(type(obj), Base)
#     assert type(obj).getvalue(obj) == 11


if __name__ == '__main__':
    main()
