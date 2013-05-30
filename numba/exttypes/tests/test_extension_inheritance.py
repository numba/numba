"""
Test Python- and Numba-level inheritance.
"""

import numba
from numba import *
from numba.testing.test_support import parametrize, main

if not numba.PY3:
    # The operation is valid in Python 3

    __doc__ = """
    >>> Base.py_method(object())
    Traceback (most recent call last):
    ...
    TypeError: unbound method numba_function_or_method object must be called with Base instance as first argument (got object instance instead)
    """

def format_str(msg, *values):
    return msg % values

def make_base(compiler):
    @compiler
    class BaseClass(object):

        @void(double)
        def __init__(self, value):
            self.value = value

        @double()
        def getvalue(self):
            "Return value"
            return self.value

        @void(double)
        def setvalue(self, value):
            "Set value"
            self.value = value

        @double()
        def method(self):
            return self.getvalue()

        @double()
        def py_method(self):
            return self.value

        @object_()
        def __repr__(self):
            return format_str('Base(%s)', self.value)

    return BaseClass

def make_derived(compiler):
    BaseClass = make_base(compiler)

    @compiler
    class DerivedClass(BaseClass):

        @void(double)
        def __init__(self, value):
            self.value = value
            self.value2 = 2.0

        @double()
        def getvalue(self):
            "Return value"
            return self.value * self.value2

        @void(double)
        def setvalue2(self, value2):
            "Set value"
            self.value2 = value2

        @object_()
        def __repr__(self):
            return format_str('Derived(%s)', self.value)

    return BaseClass, DerivedClass

#------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------

@parametrize(jit, autojit)
def test_baseclass(compiler):
    Base = make_base(compiler)

    assert str(Base(10.0)) == 'Base(10.0)'
    assert Base(10.0).py_method() == 10.0

    assert Base(4.0).method() == 4.0
    assert Base(4.0).getvalue() == 4.0

    try:
        Base.py_method(object())
    except TypeError as e:
        assert e.args[0] == ('unbound method numba_function_or_method '
                             'object must be called with BaseClass '
                             'instance as first argument (got object '
                             'instance instead)'), e.args[0]
    else:
        raise Exception("Expected an exception")

@parametrize(jit) #, autojit)
def test_derivedclass(compiler):
    Base, Derived = make_derived(compiler)

    assert str(Derived(20.0)) == 'Derived(20.0)'
    assert Derived(10.0).py_method() == 10.0

    assert Derived(4.0).method() == 8.0
    assert Derived(4.0).getvalue() == 8.0

    obj = Derived(4.0)
    obj.value2 = 3.0
    result = obj.method()
    assert result == 12.0, result


if __name__ == '__main__':
    main()
