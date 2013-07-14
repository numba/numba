import dis
import itertools
from pprint import pprint
from numbapro.npm2.symbolic import SymbolicExecution
from numbapro.npm2.typing import Infer
from numbapro.npm2 import types, functions
from numbapro.npm2.codegen import CodeGen, ImpLib
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

    return sum

@testcase
def test_codegen():
    dis.dis(foo)
    se = SymbolicExecution(foo)
    se.interpret()

    for blk in se.blocks:
        print blk

    funclib = functions.get_builtin_function_library()

    args = {'a': types.int32, 'b': types.int32}
    return_type = types.int32

    infer = Infer(func = se.func,
                  blocks = se.blocks,
                  args = args,
                  return_type = return_type,
                  funclib = funclib)
    infer.infer()

    for blk in se.blocks:
        print blk


    implib = ImpLib(funclib)
    implib.populate_builtin()

    cg = CodeGen(func = se.func,
                 blocks = se.blocks,
                 args = args,
                 return_type = return_type,
                 implib = implib)

    cg.codegen()

    print cg.lmod

    from llvm import passes as lp
    from llvm import ee as le
    eb = le.EngineBuilder.new(cg.lmod)
    tm = eb.select_target()
    pms = lp.build_pass_managers(tm, mod=cg.lmod, opt=3)
    pms.pm.run(cg.lmod)
    print cg.lmod

if __name__ == '__main__':
    main()
