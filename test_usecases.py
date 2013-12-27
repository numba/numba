from __future__ import print_function
from pprint import pprint
from numba import (bytecode, interpreter, typeinfer, typing, types,
                   lowering, targets)
from numba.tests import usecases


def main():
    bc = bytecode.ByteCode(func=usecases.andor)
    # bc = bytecode.ByteCode(func=usecases.redefine1)
    # bc = bytecode.ByteCode(func=usecases.sum1d)
    print(bc.dump())

    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()
    interp.dump()

    for syn in interp.syntax_info:
        print(syn)

    interp.verify()

    ctx = typing.Context()
    infer = typeinfer.TypeInferer(ctx, interp.blocks)
    infer.seed_type('x', types.int32)
    infer.seed_type('y', types.int32)
    # infer.seed_type('s', types.int32)
    # infer.seed_type('e', types.int32)
    # infer.seed_return(types.int32)
    infer.dump()

    infer.build_constrain()
    infer.dump()
    infer.propagate()

    typemap, restype, calltypes = infer.unify()
    print(calltypes)

    pprint(typemap)
    pprint(restype)

    fndesc = lowering.describe_function(interp, typemap, restype, calltypes)

    cpuctx = targets.CPUContext()
    lower = lowering.Lower(cpuctx, fndesc)
    lower.lower()
    cpuctx.optimize(lower.module)
    print(lower.module)



if __name__ == '__main__':
    main()
