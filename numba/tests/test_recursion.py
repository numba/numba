from numba import *
import numba as nb

#------------------------------------------------------------------------
# Jit function recursion
#------------------------------------------------------------------------

@jit(int_(int_))
def fac(arg):
    if arg == 1:
        return 1
    else:
        return arg * fac(arg - 1)

assert fac(10) == fac.py_func(10)

#------------------------------------------------------------------------
# Autojit recursion
#------------------------------------------------------------------------

# TODO: support recursion for autojit

@autojit
def fac2(arg):
    if arg == 1:
        return 1
    else:
        return arg * fac2(arg - 1)

#assert fac2(10) == fac2.py_func(10)

#------------------------------------------------------------------------
# Extension type recursion
#------------------------------------------------------------------------

@jit
class SimpleClass(object):

    @void(int_)
    def __init__(self, value):
        self.value = value

    @int_(int_)
    def fac(self, value):
        if value == 1:
            return self.value
        else:
            return value * self.fac(value - 1)

obj = SimpleClass(1)
assert obj.fac(10) == fac.py_func(10)

# ______________________________________________________________________

@jit
class ToughClass(object):

    @void(int_)
    def __init__(self, value):
        self.value = value

    @int_(int_)
    def func1(self, value):
        return self.func2(value + self.value)

    @int_(int_)
    def func2(self, value):
        return self.func3(value + self.value)

    @int_(int_)
    def func3(self, value):
        if value < 5:
            return self.func1(value + self.value)

        return value


obj = ToughClass(1)
assert obj.func1(1) == 6

# ______________________________________________________________________
