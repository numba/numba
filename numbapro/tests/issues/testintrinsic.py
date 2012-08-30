from llvm.core import *
from llvm.ee import *
from ctypes import *

module = Module.new('testintrinsic')
ee = ExecutionEngine.new(module)

def error(got, expect):
    return abs(got - expect)/(expect + 1e-31)

def test_powi(ty, cty, x, y):
    func = module.add_function(Type.function(ty, [ty, Type.int()]), name='powi_test_%s'%ty)
    bldr = Builder.new(func.append_basic_block('entry'))

    power = Function.intrinsic(module, INTR_POWI, [ty, Type.int()])
    result = bldr.call(power, [func.args[0], func.args[1]])
    bldr.ret(result)

    print(func)
    func.verify()

    fptr = ee.get_pointer_to_function(func)
    FUNC_TYPE = CFUNCTYPE(cty, cty, c_int)

    testunit = FUNC_TYPE(fptr)

    got = testunit(x, y)
    expect = x ** y

    assert error(got, expect) < 1e-6


def test_pow(ty, cty, x, y):
    func = module.add_function(Type.function(ty, [ty, ty]), name='pow_test_%s'%ty)
    bldr = Builder.new(func.append_basic_block('entry'))

    power = Function.intrinsic(module, INTR_POW, [ty])
    result = bldr.call(power, [func.args[0], func.args[1]])
    bldr.ret(result)

    print(func)
    func.verify()

    fptr = ee.get_pointer_to_function(func)
    FUNC_TYPE = CFUNCTYPE(cty, cty, cty)

    testunit = FUNC_TYPE(fptr)

    got = testunit(x, y)
    expect = x ** y

    assert error(got, expect) < 1e-6


test_powi(Type.float(), c_float, 0.12, 7)
test_powi(Type.double(), c_double, 0.12, 7)

test_pow(Type.float(), c_float, 0.12, 7)
test_pow(Type.double(), c_double, 0.12, 7)

print 'All good!'

