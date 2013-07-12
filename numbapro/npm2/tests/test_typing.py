import dis
import itertools
from pprint import pprint
from numbapro.npm2.symbolic import SymbolicExecution
from numbapro.npm2.typing import Infer, Type
from numbapro.npm2 import types
from .support import testcase, main


def foo(a, b):
    sum = a
    sum = b
    if a == b:
        i = 0
        i = 1
        for i in range(b):
            sum += i
            i = 132
            i = 321
        i = 2
    return sum

@testcase
def test_type_coercion():
    signed = types.int8, types.int16, types.int32, types.int64
    unsigned = types.uint8, types.uint16, types.uint32, types.uint64
    real = types.float32, types.float64
    complex = types.complex64, types.complex128

    numerics = signed + unsigned + real + complex

    for fromty, toty in itertools.product(numerics, numerics):
        pts = Type(fromty).coerce(Type(toty), noraise=True)
        print '%s -> %s :: %s' % (fromty, toty, pts)

@testcase
def test_infer():
    dis.dis(foo)
    se = SymbolicExecution(foo)
    se.interpret()

    for blk in se.blocks:
        print blk

    infer = Infer(symbolic = se,
                  args = {'a': types.int32, 'b': types.int32},
                  return_type = types.int32,
                  intp = tuple.__itemsize__)
    typemap = infer.infer()
    pprint(typemap)




if __name__ == '__main__':
    main()
