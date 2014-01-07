from __future__ import print_function
from pprint import pprint
from numba import (bytecode, interpreter, typing, typeinfer, lowering, targets,
                   irpasses)
from numba import DEBUG


class Flags(object):
    FLAGS_SET = frozenset(['enable_pyobject',
                           'force_pyobject'])

    def __init__(self):
        self._enabled = set()

    def set(self, name):
        if name not in self.FLAGS_SET:
            raise NameError("Invalid flag: %s" % name)
        self._enabled.add(name)

    def unset(self, name):
        if name not in self.FLAGS_SET:
            raise NameError("Invalid flag: %s" % name)
        self._enabled.discard(name)

    def __getattr__(self, name):
        if name not in self.FLAGS_SET:
            raise NameError("Invalid flag: %s" % name)
        return name in self._enabled

    def __repr__(self):
        return "Flags(%s)" % ', '.join(str(x) for x in self._enabled)

DEFAULT_FLAGS = Flags()


def compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS):
    """
    Compile the function is an isolated environment.
    Good for testing.
    """
    typingctx = typing.Context()
    targetctx = targets.CPUContext()
    cfunc, whyinferfailed = compile_extra(typingctx, targetctx, func, args,
                                          return_type, flags)
    return targetctx, cfunc, whyinferfailed


def compile_extra(typingctx, targetctx, func, args, return_type, flags):
    """
    Args
    ----
    - return_type
        Use ``None`` to indicate
    """
    # Translate to IR
    interp = translate_stage(func)
    # Optimize
    ir_optimize_stage(interp)

    if not flags.force_pyobject:
        try:
            # Type inference
            typemap, restype, calltypes = type_inference_stage(typingctx, interp,
                                                               args, return_type)
        except Exception, fail_reason:
            if not flags.enable_pyobject:
                raise

            func = py_lowering_stage(targetctx, interp)
        else:
            fail_reason = None
            func = native_lowering_stage(targetctx, interp, typemap, restype,
                                         calltypes)

    else:
        # Forced to use all python mode
        func = py_lowering_stage(targetctx, interp)
        fail_reason = None

    return func, fail_reason


def translate_stage(func):
    bc = bytecode.ByteCode(func=func)
    if DEBUG:
        print(bc.dump())

    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()

    if DEBUG:
        interp.dump()
        for syn in interp.syntax_info:
            print(syn)

    interp.verify()
    return interp


def ir_optimize_stage(interp):
    irpasses.RemoveRedundantAssign(interp).run()
    if DEBUG:
        interp.dump()


def type_inference_stage(typingctx, interp, args, return_type):
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

    if DEBUG:
        pprint(typemap)
        pprint(restype)
        pprint(calltypes)

    return typemap, restype, calltypes


def native_lowering_stage(targetctx, interp, typemap, restype, calltypes):
    # Lowering
    fndesc = lowering.describe_function(interp, typemap, restype, calltypes)

    lower = lowering.Lower(targetctx, fndesc)
    lower.lower()
    
    # Prepare for execution
    cfunc = targetctx.get_executable(lower.function, fndesc)

    targetctx.insert_user_function(cfunc, fndesc)

    return cfunc


def py_lowering_stage(targetctx, interp):
    fndesc = lowering.describe_pyfunction(interp)
    lower = lowering.PyLower(targetctx, fndesc)
    lower.lower()

    if DEBUG:
        print(lower.module)

    cfunc = targetctx.get_executable(lower.function, fndesc)

    return cfunc
