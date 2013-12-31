from __future__ import print_function
from pprint import pprint
from numba import (bytecode, interpreter, typing, typeinfer, lowering,
                   targets )


def compile_isolated(func, args, return_type=None):
    """
    Compile the function is an isolated environment.
    Good for testing.
    """
    typingctx = typing.Context()
    targetctx = targets.CPUContext()
    cfunc = compile_extra(typingctx, targetctx, func, args, return_type)
    return targetctx, cfunc


def compile_extra(typingctx, targetctx, func, args, return_type=None):
    # Translate to IR
    bc = bytecode.ByteCode(func=func)
    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()
    interp.dump()

    if __debug__:
        for syn in interp.syntax_info:
            print(syn)

    interp.verify()

    # Type inference
    infer = typeinfer.TypeInferer(typingctx, interp.blocks)

    # Seed argument types
    for arg, ty in zip(interp.argspec.args, args):
        infer.seed_type(arg, ty)

    # Seed return type
    if return_type is not None:
        infer.seed_return(return_type)

    infer.build_constrain()
    infer.propagate()
    typemap, restype, calltypes = infer.unify()

    if __debug__:
        pprint(typemap)
        pprint(restype)
        pprint(calltypes)

    # Lowering
    fndesc = lowering.describe_function(interp, typemap, restype, calltypes)

    lower = lowering.Lower(targetctx, fndesc)
    lower.lower()

    if __debug__:
        print(lower.module)

    cfunc = targetctx.get_executable(lower.function, fndesc)
    return cfunc