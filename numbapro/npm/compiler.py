import inspect
from . import symbolic, typing, codegen, execution

def compile(func, retty, argtys):
    # symbolic interpretation
    se = symbolic.SymbolicExecution(func)
    se.visit()
    #print se.dump()

    argspec = inspect.getargspec(func)
    assert not argspec.keywords
    assert not argspec.varargs
    assert not argspec.defaults

    # type infernece
    tydict = dict(zip(argspec.args, argtys))
    tydict[''] = retty

    addrsize = tuple.__itemsize__ * 8
    globals = func.func_globals
    infer = typing.Infer(se.blocks, tydict, globals, intp=addrsize)

    typemap = infer.infer()

    #pprint.pprint(typemap)

    # code generation
    name = get_func_name(func)
    cg = codegen.CodeGen(name, se.blocks, typemap, globals,
                         argtys, retty, intp=addrsize)
    lfunc = cg.generate()
    gvars = cg.extern_globals
    # execution
    return execution.JIT(lfunc, retty, argtys, gvars, globals)

def get_func_name(func):
    try:
        return func.func_name
    except AttributeError:
        try:
            return func.__name__
        except AttributeError:
            return str(func)

