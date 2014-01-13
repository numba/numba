from __future__ import print_function
from pprint import pprint
from collections import namedtuple
from numba import (bytecode, interpreter, typing, typeinfer, lowering, targets,
                   irpasses, utils, config)


class Flags(utils.ConfigOptions):
    OPTIONS = frozenset(['enable_pyobject',
                         'force_pyobject'])


DEFAULT_FLAGS = Flags()


CR_FIELDS = ["typing_context",
             "target_context",
             "entry_point",
             "entry_point_addr",
             "typing_error",
             "llvm_func",
             "argtypes",]


CompileResult = namedtuple("CompileResult", CR_FIELDS)


def compile_result(**kws):
    keys = set(kws.keys())
    fieldset = set(CR_FIELDS)
    badnames = keys - fieldset
    if badnames:
        raise NameError(*basenames)
    missing = fieldset - keys
    for k in missing:
        kws[k] = None
    return CompileResult(**kws)


def compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS):
    """
    Compile the function is an isolated environment.
    Good for testing.
    """
    typingctx = typing.Context()
    targetctx = targets.CPUContext()
    return compile_extra(typingctx, targetctx, func, args, return_type, flags)


def compile_extra(typingctx, targetctx, func, args, return_type, flags):
    """
    Args
    ----
    - return_type
        Use ``None`` to indicate
    """
    # Translate to IR
    interp = translate_stage(func)

    if not flags.force_pyobject:
        try:
            # Type inference
            typemap, restype, calltypes = type_inference_stage(typingctx, interp,
                                                               args, return_type)
        except Exception, fail_reason:
            if not flags.enable_pyobject:
                raise

            func, fnptr, lfunc = py_lowering_stage(targetctx, interp)
        else:
            fail_reason = None
            func, fnptr, lfunc = native_lowering_stage(targetctx, interp,
                                                       typemap, restype,
                                                       calltypes)

    else:
        # Forced to use all python mode
        func, fnptr, lfunc = py_lowering_stage(targetctx, interp)
        fail_reason = None

    cr = compile_result(typing_context=typingctx,
                        target_context=targetctx,
                        entry_point=func,
                        entry_point_addr=fnptr,
                        typing_error=fail_reason,
                        llvm_func=lfunc,
                        argtypes=tuple(args))
    return cr


def translate_stage(func):
    bc = bytecode.ByteCode(func=func)
    if config.DEBUG:
        print(bc.dump())

    interp = interpreter.Interpreter(bytecode=bc)
    interp.interpret()

    if config.DEBUG:
        interp.dump()
        for syn in interp.syntax_info:
            print(syn)

    interp.verify()
    return interp


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

    if config.DEBUG:
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
    cfunc, fnptr = targetctx.get_executable(lower.function, fndesc)

    targetctx.insert_user_function(cfunc, fndesc)

    return cfunc, fnptr, lower.function


def py_lowering_stage(targetctx, interp):
    # Optimize for python code
    ir_optimize_for_py_stage(interp)

    fndesc = lowering.describe_pyfunction(interp)
    lower = lowering.PyLower(targetctx, fndesc)
    lower.lower()

    if config.DEBUG:
        print(lower.module)

    cfunc, fnptr = targetctx.get_executable(lower.function, fndesc)

    return cfunc, fnptr, lower.function


def ir_optimize_for_py_stage(interp):
    """
    This passes breaks semantic for the type inferer but they reduces
    refct calls for object mode.
    """
    irpasses.RemoveRedundantAssign(interp).run()
    if config.DEBUG:
        print("ir optimize".center(80, '-'))
        interp.dump()
