from __future__ import print_function, division, absolute_import
from pprint import pprint
from contextlib import contextmanager
from collections import namedtuple, defaultdict

from numba import (bytecode, interpreter, typing, typeinfer, lowering,
                   objmode, irpasses, utils, config, type_annotations,
                   types, ir, assume, looplifting, macro)
from numba.targets import cpu


class Flags(utils.ConfigOptions):
    # These options are all false by default, but the defaults are
    # different with the @jit decorator (see targets.options.TargetOptions).

    OPTIONS = frozenset([
        # Enable loop-lifting
        'enable_looplift',
        # Enable pyobject mode (in general)
        'enable_pyobject',
        # Enable pyobject mode inside lifted loops
        'enable_pyobject_looplift',
        # Force pyobject mode inside the whole function
        'force_pyobject',
        'no_compile',
        'no_wraparound',
        'boundcheck',
        ])


DEFAULT_FLAGS = Flags()


CR_FIELDS = ["typing_context",
             "target_context",
             "entry_point",
             "typing_error",
             "type_annotation",
             "llvm_module",
             "llvm_func",
             "signature",
             "objectmode",
             "lifted",
             "fndesc"]


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


class _CompileStatus(object):
    """
    Used like a C record
    """
    __slots__ = 'fail_reason', 'use_python_mode', 'can_fallback'


@contextmanager
def _fallback_context(status):
    try:
        yield
    except Exception as e:
        if not status.can_fallback:
            raise
        if utils.PYVERSION >= (3,):
            # Clear all references attached to the traceback
            e = e.with_traceback(None)
        status.fail_reason = e
        status.use_python_mode = True


def compile_extra(typingctx, targetctx, func, args, return_type, flags,
                  locals):
    """
    Args
    ----
    - return_type
        Use ``None`` to indicate
    """
    bc = bytecode.ByteCode(func=func)
    if config.DUMP_BYTECODE:
        print(bc.dump())
    return compile_bytecode(typingctx, targetctx, bc, args,
                            return_type, flags, locals)


def compile_bytecode(typingctx, targetctx, bc, args, return_type, flags,
                     locals, lifted=()):
    interp = translate_stage(bc)
    nargs = len(interp.argspec.args)
    if len(args) > nargs:
        raise TypeError("Too many argument types")

    status = _CompileStatus()
    status.can_fallback = flags.enable_pyobject
    status.fail_reason = None
    status.use_python_mode = flags.force_pyobject

    targetctx = targetctx.localized()
    targetctx.metadata['wraparound'] = not flags.no_wraparound

    if not status.use_python_mode:
        with _fallback_context(status):
            legalize_given_types(args, return_type)
            # Type inference
            typemap, return_type, calltypes = type_inference_stage(typingctx,
                                                                   interp,
                                                                   args,
                                                                   return_type,
                                                                   locals)

        if not status.use_python_mode:
            with _fallback_context(status):
                legalize_return_type(return_type, interp, targetctx)

    if status.use_python_mode and flags.enable_looplift:
        assert not lifted

        # Try loop lifting
        loop_flags = flags.copy()
        outer_flags = flags.copy()
        # Do not recursively loop lift
        outer_flags.unset('enable_looplift')
        loop_flags.unset('enable_looplift')
        if not flags.enable_pyobject_looplift:
            loop_flags.unset('enable_pyobject')

        def dispatcher_factory(loopbc):
            from . import dispatcher
            return dispatcher.LiftedLoop(loopbc, typingctx, targetctx,
                                         locals, loop_flags)

        entry, loops = looplifting.lift_loop(bc, dispatcher_factory)
        if loops:
            # Some loops were extracted
            cres = compile_bytecode(typingctx, targetctx, entry, args,
                                    return_type, outer_flags, locals,
                                    lifted=tuple(loops))
            return cres

    if status.use_python_mode:
        # Object mode compilation
        func, lmod, lfunc, fndesc = py_lowering_stage(targetctx, interp,
                                                             flags.no_compile)
        typemap = defaultdict(lambda: types.pyobject)
        calltypes = defaultdict(lambda: types.pyobject)

        return_type = types.pyobject

        if len(args) != nargs:
            # append missing
            args = tuple(args) + (types.pyobject,) * (nargs - len(args))
    else:
        # Native mode compilation
        func, lmod, lfunc, fndesc = native_lowering_stage(targetctx,
                                                                 interp,
                                                                 typemap,
                                                                 return_type,
                                                                 calltypes,
                                                                 flags.no_compile)

    type_annotation = type_annotations.TypeAnnotation(interp=interp,
                                                      typemap=typemap,
                                                      calltypes=calltypes,
                                                      lifted=lifted)
    if config.ANNOTATE:
        print("ANNOTATION".center(80, '-'))
        print(type_annotation)
        print('=' * 80)

    signature = typing.signature(return_type, *args)

    assert lfunc.module is lmod
    cr = compile_result(typing_context=typingctx,
                        target_context=targetctx,
                        entry_point=func,
                        typing_error=status.fail_reason,
                        type_annotation=type_annotation,
                        llvm_func=lfunc,
                        llvm_module=lmod,
                        signature=signature,
                        objectmode=status.use_python_mode,
                        lifted=lifted,
                        fndesc=fndesc,)
    return cr


def _is_nopython_types(t):
    return t != types.pyobject and not isinstance(t, types.Dummy)


def legalize_given_types(args, return_type):
    # Filter argument types
    for i, a in enumerate(args):
        if not _is_nopython_types(a):
            raise TypeError("Arg %d of %s is not legal in nopython "
                            "mode" % (i, a))
    # Filter return type
    if (return_type and return_type != types.none and
            not _is_nopython_types(return_type)):
        raise TypeError('Return type of %s is not legal in nopython '
                        'mode' % (return_type,))


def legalize_return_type(return_type, interp, targetctx):
    """
    Only accept array return type iff it is passed into the function.
    """
    assert assume.return_argument_array_only

    if not isinstance(return_type, types.Array):
        return

    # Walk IR to discover all return statements
    retstmts = []
    for bid, blk in interp.blocks.items():
        for inst in blk.body:
            if isinstance(inst, ir.Return):
                retstmts.append(inst)

    assert retstmts, "No return statemants?"

    # FIXME: In the future, we can return an array that is either a dynamically
    #        allocated array or an array that is passed as argument.  This
    #        must be statically resolvable.

    # The return value must be the first modification of the value.
    arguments = frozenset("%s.1" % arg for arg in interp.argspec.args)

    for ret in retstmts:
        if ret.value.name not in arguments:
            raise TypeError("Only accept returning of array passed into the "
                            "function as argument")

    # Legalized; tag return handling
    targetctx.metadata['return.array'] = 'arg'


def translate_stage(bytecode):
    interp = interpreter.Interpreter(bytecode=bytecode)

    if config.DUMP_CFG:
        interp.cfa.dump()

    interp.interpret()

    if config.DEBUG:
        interp.dump()
        for syn in interp.syntax_info:
            print(syn)

    interp.verify()
    macro.expand_macros(interp.blocks)

    if config.DUMP_IR:
        interp.dump()
        for syn in interp.syntax_info:
            print(syn)

    return interp


def type_inference_stage(typingctx, interp, args, return_type, locals={}):
    if len(args) != len(interp.argspec.args):
        raise TypeError("Mismatch number of argument types")

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
    fndesc = lowering.PythonFunctionDescriptor.from_specialized_function(
        interp, typemap, restype, calltypes, mangler=targetctx.mangler)

    lower = lowering.Lower(targetctx, fndesc)
    lower.lower()

    if nocompile:
        return None, lower.module, lower.function, fndesc
    else:
        # Prepare for execution
        cfunc = targetctx.get_executable(lower.function, fndesc, lower.env)

        targetctx.insert_user_function(cfunc, fndesc)

        return cfunc, lower.module, lower.function, fndesc


def py_lowering_stage(targetctx, interp, nocompile):
    fndesc = lowering.PythonFunctionDescriptor.from_object_mode_function(interp)
    lower = objmode.PyLower(targetctx, fndesc)
    lower.lower()

    if nocompile:
        return None, lower.module, lower.function, fndesc
    else:
        cfunc = targetctx.get_executable(lower.function, fndesc, lower.env)
        return cfunc, lower.module, lower.function, fndesc


def ir_optimize_for_py_stage(interp):
    """
    This passes breaks semantic for the type inferer but they reduces
    refct calls for object mode.
    """
    irpasses.RemoveRedundantAssign(interp).run()
    if config.DEBUG:
        print("ir optimize".center(80, '-'))
        interp.dump()
