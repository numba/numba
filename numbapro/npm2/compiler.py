import inspect
from . import symbolic, typing, codegen, execution, functions

def compile(func, retty, argtys):
    # preparation
    argspec = inspect.getargspec(func)
    assert not argspec.defaults
    assert not argspec.keywords
    assert not argspec.varargs

    args = dict((arg, typ) for arg, typ in zip(argspec.args, argtys))
    return_type = retty

    funclib = functions.get_builtin_function_library()

    implib = codegen.ImpLib(funclib)
    implib.populate_builtin()

    # compilation
    blocks =  symbolic_interpret(func)
    type_infer(func, blocks, return_type, args, funclib)
    lmod, lfunc = code_generation(func, blocks, return_type, args, implib)

    jit = execution.JIT(lfunc = lfunc,
                        retty = retty,
                        argtys = argtys)
    return jit


#----------------------------------------------------------------------------
# Internals

def symbolic_interpret(func):
    se = symbolic.SymbolicExecution(func)
    se.interpret()
    return se.blocks

def type_infer(func, blocks, return_type, args, funclib):
    infer = typing.Infer(func        = func,
                         blocks      = blocks,
                         args        = args,
                         return_type = return_type,
                         funclib     = funclib)
    infer.infer()

def code_generation(func, blocks, return_type, args, implib):
    cg = codegen.CodeGen(func        = func,
                         blocks      = blocks,
                         args        = args,
                         return_type = return_type,
                         implib      = implib)
    cg.codegen()
    return cg.lmod, cg.lfunc
