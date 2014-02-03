import ctypes

import numba
from numba import *
from numba.testing.test_support import parametrize, main

def format_str(msg, *values):
    return msg % values

def make_myextension(compiler):
    @compiler
    class MyExtension(object):

        @void(double)
        def __init__(self, myfloat):
            self.value = myfloat

        @double()
        def getvalue(self):
            "Return value"
            return self.value

        @void(double)
        def setvalue(self, value):
            "Set value"
            self.value = value

        @object_()
        def __repr__(self):
            return format_str('MyExtension%s', self.value)

    return MyExtension

def make_obj_extension(compiler):
    @compiler
    class ObjectAttrExtension(object):

        def __init__(self, value1, value2):
            self.value1 = object_(value1)
            self.value2 = double(value2)

        @object_()
        def getvalue(self):
            "Return value"
            return self.value1

        @void(double)
        def setvalue(self, value):
            "Set value"
            self.value1 = value

        @object_()
        def method(self):
            return self.getvalue()

        @object_(int32)
        def method2(self, new_value):
            self.setvalue(new_value * 2)
            result = self.method()
            return result

    return ObjectAttrExtension

def make_extattr_extension():
    ObjectAttrExtension = make_obj_extension(jit)

    exttype = ObjectAttrExtension.exttype

    @jit
    class ExtensionTypeAsAttribute(object):

        def __init__(self, attr):
            self.attr = exttype(attr)

    return ExtensionTypeAsAttribute

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

@parametrize(jit, autojit)
def test_extension(compiler):
    MyExtension = make_myextension(compiler)
    # ______________________________________________________________________
    # Test methods and attributes

    obj = MyExtension(10.0)
    assert obj.value == 10.0
    assert obj._numba_attrs.value == 10.0

    obj.setvalue(20.0)
    assert obj.getvalue() == 20.0
    assert obj.value == 20.0

    obj._numba_attrs._fields_ == [('value', ctypes.c_double)]

    # ______________________________________________________________________
    # Test stringifications

    assert obj.getvalue.__name__ == 'getvalue'
    assert obj.getvalue.__doc__ == 'Return value'

    strmethod = str(type(obj.getvalue.__func__))
    if numba.PY3:
        assert strmethod == "<class 'numba_function_or_method'>"
    else:
        assert strmethod == "<type 'numba_function_or_method'>"

    return MyExtension

@parametrize(jit, autojit)
def test_obj_attributes(compiler):
    MyExtension = make_myextension(compiler)
    ObjectAttrExtension = make_obj_extension(compiler)

    # TODO: Disallow string <-> real coercions! These are conversions!
    # try:
    #     obj = ObjectAttrExtension(10.0, 'blah')
    # except TypeError as e:
    #     assert e.args[0] == 'a float is required'
    # else:
    #     raise Exception

    assert ObjectAttrExtension(10.0, 3.5).value1 == 10.0

    obj = ObjectAttrExtension('hello', 9.3)
    assert obj.value1 == 'hello'
    obj.setvalue(20.0)
    assert obj.getvalue() == 20.0

    obj.value1 = MyExtension(10.0)
    assert str(obj.value1) == "MyExtension10.0"
    assert str(obj.getvalue()) == "MyExtension10.0"
    assert str(obj.method()) == "MyExtension10.0"

    assert obj.method2(15.0) == 30.0

@parametrize(jit)
def test_extension_attribute(compiler):
    ExtensionTypeAsAttribute = make_extattr_extension()
    assert (str(ExtensionTypeAsAttribute.exttype) ==
            ("<JitExtension ExtensionTypeAsAttribute("
             "{'attr': <JitExtension ObjectAttrExtension>})>"))

if __name__ == '__main__':
    main()
