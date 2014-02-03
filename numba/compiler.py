from __future__ import print_function, division, absolute_import
from pprint import pprint
from collections import namedtuple, defaultdict
from numba import (bytecode, interpreter, typing, typeinfer, lowering,
                   irpasses, utils, config, type_annotations, types)
from numba.targets import cpu


class Flags(utils.ConfigOptions):
    OPTIONS = frozenset(['enable_pyobject',
                         'force_pyobject',
                         'no_compile'])


DEFAULT_FLAGS = Flags()


CR_FIELDS = ["typing_context",
             "target_context",
             "entry_point",
             "entry_point_addr",
             "typing_error",
             "type_annotation",
             "llvm_module",
             "llvm_func",
             "signature",
             "objectmode",]


CompileResult = namedtuple("CompileResult", CR_FIELDS)


def compile_result(**kws):
    keys = set(kws.keys())
    fieldset = set(CR_FIELDS)
    badnames = keys - fieldset
    if badnames:
        raise NameError(*badnames)
    missing = fieldset - keys
    for k in missing:
        kws[k] = None
    return CompileResult(**kws)


def compile_isolated(func, args, return_type=None, flags=DEFAULT_FLAGS,
                     locals={}):
    """
    Compile the function is an isolated environment.
    Good for testing.
    """
    typingctx = typing.Context()
    targetctx = cpu.CPUContext(typingctx)
    return compile_extra(typingctx, targetctx, func, args, return_type, flags,
                         locals)


def compile_extra(typingctx, targetctx, func, args, return_type, flags,
                  locals):
    """
    Args
    ----
    - return_type
        Use ``None`` to indicate
    """
    # Translate to IR
    interp = translate_stage(func)
    nargs = len(interp.argspec.args)

    fail_reason = None
    use_python_mode = False

    if not flags.force_pyobject:
        try:
            # Type inference
            typemap, return_type, calltypes = type_inference_stage(typingctx,
                                                                   interp,
                                                                   args,
                                                                   return_type,
                                                                   locals)
        except Exception as e:
            if not flags.enable_pyobject:
                raise

            fail_reason = e
            use_python_mode = True
    else:
        # Forced to use all python mode
        use_python_mode = True

    if use_python_mode:
        func, fnptr, lmod, lfunc = py_lowering_stage(targetctx, interp,
                                                     flags.no_compile)
        typemap = defaultdict(lambda: types.pyobject)
        calltypes = defaultdict(lambda: types.pyobject)

        return_type = types.pyobject
        args = [types.pyobject] * nargs
    else:
        func, fnptr, lmod, lfunc = native_lowering_stage(targetctx, interp,
                                                         typemap,
                                                         return_type,
                                                         calltypes,
                                                         flags.no_compile)

    type_annotation = type_annotations.TypeAnnotation(interp=interp,
                                                      typemap=typemap,
                                                      calltypes=calltypes)


    signature = typing.signature(return_type, *args)

    assert lfunc.module is lmod
    cr = compile_result(typing_context=typingctx,
                        target_context=targetctx,
                        entry_point=func,
                        entry_point_addr=fnptr,
                        typing_error=fail_reason,
                        type_annotation=type_annotation,
                        llvm_func=lfunc,
                        llvm_module=lmod,
                        signature=signature,
                        objectmode=use_python_mode)
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


def type_inference_stage(typingctx, interp, args, return_type, locals={}):
    infer = typeinfer.TypeInferer(typingctx, interp.blocks)

    # Seed argument types
    for arg, ty in zip(interp.argspec.args, args):
        infer.seed_type(arg, ty)

    # Seed return type
    if return_type is not None:
        infer.seed_return(return_type)

    # Seed local types
    for k, v in locals.items():
        infer.seed_type(k, v)

    infer.build_constrain()
    infer.propagate()
    typemap, restype, calltypes = infer.unify()

    if config.DEBUG:
        pprint(typemap)
        pprint(restype)
        pprint(calltypes)

    return typemap, restype, calltypes


def native_lowering_stage(targetctx, interp, typemap, restype, calltypes,
                          nocompile):
    # Lowering
    fndesc = lowering.describe_function(interp, typemap, restype, calltypes)

    lower = lowering.Lower(targetctx, fndesc)
    lower.lower()

    if nocompile:
        return None, 0, lower.module, lower.function
    else:
        # Prepare for execution
        cfunc, fnptr = targetctx.get_executable(lower.function, fndesc)

        targetctx.insert_user_function(cfunc, fndesc)

        return cfunc, fnptr, lower.module, lower.function


def py_lowering_stage(targetctx, interp, nocompile):
    # Optimize for python code
    ir_optimize_for_py_stage(interp)

    fndesc = lowering.describe_pyfunction(interp)
    lower = lowering.PyLower(targetctx, fndesc)
    lower.lower()

    if config.DEBUG:
        print(lower.module)

    if nocompile:
        return None, 0, lower.module, lower.function
    else:
        cfunc, fnptr = targetctx.get_executable(lower.function, fndesc)
        return cfunc, fnptr, lower.module, lower.function


def ir_optimize_for_py_stage(interp):
    """
    This passes breaks semantic for the type inferer but they reduces
    refct calls for object mode.
    """
    irpasses.RemoveRedundantAssign(interp).run()
    if config.DEBUG:
        print("ir optimize".center(80, '-'))
        interp.dump()
