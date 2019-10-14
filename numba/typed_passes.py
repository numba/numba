from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import warnings

from . import (config,  errors, types, rewrites, typeinfer, funcdesc, lowering,
               utils, typing, ir)

from .parfor import PreParforPass as _parfor_PreParforPass
from .parfor import ParforPass as _parfor_ParforPass
from .parfor import Parfor

from .compiler_machinery import FunctionPass, LoweringPass, register_pass
from .annotations import type_annotations
from .ir_utils import (raise_on_unsupported_feature, warn_deprecated,
                       check_and_legalize_ir, guard, dead_code_elimination,
                       simplify_CFG)


@contextmanager
def fallback_context(state, msg):
    """
    Wraps code that would signal a fallback to object mode
    """
    try:
        yield
    except Exception as e:
        if not state.status.can_fallback:
            raise
        else:
            if utils.PYVERSION >= (3,):
                # Clear all references attached to the traceback
                e = e.with_traceback(None)
            # this emits a warning containing the error message body in the
            # case of fallback from npm to objmode
            loop_lift = '' if state.flags.enable_looplift else 'OUT'
            msg_rewrite = ("\nCompilation is falling back to object mode "
                           "WITH%s looplifting enabled because %s"
                           % (loop_lift, msg))
            warnings.warn_explicit('%s due to: %s' % (msg_rewrite, e),
                                   errors.NumbaWarning,
                                   state.func_id.filename,
                                   state.func_id.firstlineno)
            raise


def type_inference_stage(typingctx, interp, args, return_type, locals={}):
    if len(args) != interp.arg_count:
        raise TypeError("Mismatch number of argument types")

    warnings = errors.WarningsFixer(errors.NumbaWarning)
    infer = typeinfer.TypeInferer(typingctx, interp, warnings)
    with typingctx.callstack.register(infer, interp.func_id, args):
        # Seed argument types
        for index, (name, ty) in enumerate(zip(interp.arg_names, args)):
            infer.seed_argument(name, index, ty)

        # Seed return type
        if return_type is not None:
            infer.seed_return(return_type)

        # Seed local types
        for k, v in locals.items():
            infer.seed_type(k, v)

        infer.build_constraint()
        infer.propagate()
        typemap, restype, calltypes = infer.unify()

    # Output all Numba warnings
    warnings.flush()

    return typemap, restype, calltypes


@register_pass(mutates_CFG=True, analysis_only=False)
class NopythonTypeInference(FunctionPass):
    _name = "nopython_type_inference"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Type inference and legalization
        """
        with fallback_context(state, 'Function "%s" failed type inference'
                              % (state.func_id.func_name,)):
            # Type inference
            typemap, return_type, calltypes = type_inference_stage(
                state.typingctx,
                state.func_ir,
                state.args,
                state.return_type,
                state.locals)
            state.typemap = typemap
            state.return_type = return_type
            state.calltypes = calltypes

        def legalize_return_type(return_type, interp, targetctx):
            """
            Only accept array return type iff it is passed into the function.
            Reject function object return types if in nopython mode.
            """
            if not targetctx.enable_nrt and isinstance(return_type, types.Array):
                # Walk IR to discover all arguments and all return statements
                retstmts = []
                caststmts = {}
                argvars = set()
                for bid, blk in interp.blocks.items():
                    for inst in blk.body:
                        if isinstance(inst, ir.Return):
                            retstmts.append(inst.value.name)
                        elif isinstance(inst, ir.Assign):
                            if (isinstance(inst.value, ir.Expr)
                                    and inst.value.op == 'cast'):
                                caststmts[inst.target.name] = inst.value
                            elif isinstance(inst.value, ir.Arg):
                                argvars.add(inst.target.name)

                assert retstmts, "No return statements?"

                for var in retstmts:
                    cast = caststmts.get(var)
                    if cast is None or cast.value.name not in argvars:
                        raise TypeError("Only accept returning of array passed into "
                                        "the function as argument")

            elif (isinstance(return_type, types.Function) or
                    isinstance(return_type, types.Phantom)):
                msg = "Can't return function object ({}) in nopython mode"
                raise TypeError(msg.format(return_type))

        with fallback_context(state, 'Function "%s" has invalid return type'
                              % (state.func_id.func_name,)):
            legalize_return_type(state.return_type, state.func_ir,
                                 state.targetctx)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class AnnotateTypes(FunctionPass):
    _name = "annotate_types"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Create type annotation after type inference
        """
        state.type_annotation = type_annotations.TypeAnnotation(
            func_ir=state.func_ir,
            typemap=state.typemap,
            calltypes=state.calltypes,
            lifted=state.lifted,
            lifted_from=state.lifted_from,
            args=state.args,
            return_type=state.return_type,
            html_output=config.HTML)

        if config.ANNOTATE:
            print("ANNOTATION".center(80, '-'))
            print(state.type_annotation)
            print('=' * 80)
        if config.HTML:
            with open(config.HTML, 'w') as fout:
                state.type_annotation.html_annotate(fout)
        return False


@register_pass(mutates_CFG=True, analysis_only=False)
class NopythonRewrites(FunctionPass):
    _name = "nopython_rewrites"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Perform any intermediate representation rewrites after type
        inference.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        assert isinstance(getattr(state, 'typemap', None), dict)
        assert isinstance(getattr(state, 'calltypes', None), dict)
        msg = ('Internal error in post-inference rewriting '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            rewrites.rewrite_registry.apply('after-inference', state)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class PreParforPass(FunctionPass):

    _name = "pre_parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        preparfor_pass = _parfor_PreParforPass(
            state.func_ir,
            state.type_annotation.typemap,
            state.type_annotation.calltypes, state.typingctx,
            state.flags.auto_parallel,
            state.parfor_diagnostics.replaced_fns
        )

        preparfor_pass.run()
        return True


# this is here so it pickles and for no other reason
def _reload_parfors():
    """Reloader for cached parfors
    """
    # Re-initialize the parallel backend when load from cache.
    from numba.npyufunc.parallel import _launch_threads
    _launch_threads()


@register_pass(mutates_CFG=True, analysis_only=False)
class ParforPass(FunctionPass):

    _name = "parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _parfor_ParforPass(state.func_ir, state.type_annotation.typemap,
                                         state.type_annotation.calltypes, state.return_type, state.typingctx,
                                         state.flags.auto_parallel, state.flags, state.parfor_diagnostics)
        parfor_pass.run()

        # check the parfor pass worked and warn if it didn't
        has_parfor = False
        for blk in state.func_ir.blocks.values():
            for stmnt in blk.body:
                if isinstance(stmnt, Parfor):
                    has_parfor = True
                    break
            else:
                continue
            break

        if not has_parfor:
            # parfor calls the compiler chain again with a string
            if not (config.DISABLE_PERFORMANCE_WARNINGS or
                    state.func_ir.loc.filename == '<string>'):
                url = ("http://numba.pydata.org/numba-doc/latest/user/"
                       "parallel.html#diagnostics")
                msg = ("\nThe keyword argument 'parallel=True' was specified "
                       "but no transformation for parallel execution was "
                       "possible.\n\nTo find out why, try turning on parallel "
                       "diagnostics, see %s for help." % url)
                warnings.warn(errors.NumbaPerformanceWarning(msg,
                                                             state.func_ir.loc))

        # Add reload function to initialize the parallel backend.
        state.reload_init.append(_reload_parfors)
        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class DumpParforDiagnostics(FunctionPass):

    _name = "dump_parfor_diagnostics"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        if state.flags.auto_parallel.enabled:
            if config.PARALLEL_DIAGNOSTICS:
                if state.parfor_diagnostics is not None:
                    state.parfor_diagnostics.dump(config.PARALLEL_DIAGNOSTICS)
                else:
                    raise RuntimeError("Diagnostics failed.")
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class NativeLowering(LoweringPass):

    _name = "native_lowering"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        targetctx = state.targetctx
        library = state.library
        interp = state.func_ir  # why is it called this?!
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata

        msg = ("Function %s failed at nopython "
               "mode lowering" % (state.func_id.func_name,))
        with fallback_context(state, msg):
            # Lowering
            fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(
                interp, typemap, restype, calltypes, mangler=targetctx.mangler,
                inline=flags.forceinline, noalias=flags.noalias)

            with targetctx.push_code_library(library):
                lower = lowering.Lower(targetctx, library, fndesc, interp,
                                       metadata=metadata)
                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)
                env = lower.env
                call_helper = lower.call_helper
                del lower

            from numba.compiler import _LowerResult  # TODO: move this
            if flags.no_compile:
                state['cr'] = _LowerResult(fndesc, call_helper, cfunc=None, env=env)
            else:
                # Prepare for execution
                cfunc = targetctx.get_executable(library, fndesc, env)
                # Insert native function for use by other jitted-functions.
                # We also register its library to allow for inlining.
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state['cr'] = _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)
        return True


@register_pass(mutates_CFG=False, analysis_only=True)
class IRLegalization(FunctionPass):

    _name = "ir_legalization"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        raise_on_unsupported_feature(state.func_ir, state.typemap)
        warn_deprecated(state.func_ir, state.typemap)
        # NOTE: this function call must go last, it checks and fixes invalid IR!
        check_and_legalize_ir(state.func_ir)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class NoPythonBackend(FunctionPass):

    _name = "nopython_backend"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        NativeLowering().run_pass(state) # TODO: Pull this out into the pipeline
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)

        from numba.compiler import compile_result
        state.cr = compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            entry_point=lowered.cfunc,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            objectmode=False,
            interpmode=False,
            lifted=state.lifted,
            fndesc=lowered.fndesc,
            environment=lowered.env,
            metadata=state.metadata,
            reload_init=state.reload_init,
        )
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineOverloads(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.extending.overload
    decorator directly into the site of its call depending on the value set in
    the 'inline' kwarg to the decorator.

    This is a typed pass. CFG simplification and DCE are performed on
    completion.
    """

    _name = "inline_overloads"

    def __init__(self):
        FunctionPass.__init__(self)

    _DEBUG = False

    def run_pass(self, state):
        """Run inlining of overloads
        """
        if config.DEBUG or self._DEBUG:
            print('before overload inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        modified = False
        work_list = list(state.func_ir.blocks.items())
        # use a work list, look for call sites via `ir.Expr.op == call` and
        # then pass these to `self._do_work` to make decisions about inlining.
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if isinstance(instr, ir.Assign):
                    expr = instr.value
                    if isinstance(expr, ir.Expr) and expr.op == 'call':
                        if guard(self._do_work, state, work_list, block, i,
                                 expr):
                            modified = True
                            break  # because block structure changed

        if config.DEBUG or self._DEBUG:
            print('after overload inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))

        if modified:
            # clean up blocks
            dead_code_elimination(state.func_ir,
                                  typemap=state.type_annotation.typemap)
            # clean up unconditional branches that appear due to inlined
            # functions introducing blocks
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        if config.DEBUG or self._DEBUG:
            print('after overload inline DCE'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))

        return True

    def _do_work(self, state, work_list, block, i, expr):
        from numba.inline_closurecall import (inline_closure_call,
                                              callee_ir_validator)

        # try and get a definition for the call, this isn't always possible as
        # it might be a eval(str)/part generated awaiting update etc. (parfors)
        to_inline = None
        try:
            to_inline = state.func_ir.get_definition(expr.func)
        except Exception:
            return False

        # do not handle closure inlining here, another pass deals with that.
        if getattr(to_inline, 'op', False) == 'make_function':
            return False

        # check this is a known and typed function
        try:
            func_ty = state.type_annotation.typemap[expr.func.name]
        except KeyError:
            # e.g. Calls to CUDA Intrinsic have no mapped type so KeyError
            return False
        if not hasattr(func_ty, 'get_call_type'):
            return False

        # search the templates for this overload looking for "inline"
        templates = getattr(func_ty, 'templates', None)
        if templates is None:
            return False

        sig = state.type_annotation.calltypes[expr]

        impl = None
        for template in templates:
            inline_type = getattr(template, '_inline', None)
            if inline_type is None:
                # inline not defined
                continue
            if not inline_type.is_never_inline:
                try:
                    impl = template._overload_func(*sig.args)
                    if impl is None:
                        raise Exception  # abort for this template
                    break
                except Exception:
                    continue
        else:
            return False

        # at this point we know we maybe want to inline something and there's
        # definitely something that could be inlined.

        do_inline = True
        if not inline_type.is_always_inline:
            from numba.typing.templates import _inline_info
            caller_inline_info = _inline_info(state.func_ir,
                                              state.type_annotation.typemap,
                                              state.type_annotation.calltypes,
                                              sig)

            # must be a cost-model function, run the function
            iinfo = template._inline_overloads[sig.args]['iinfo']
            if inline_type.has_cost_model:
                do_inline = inline_type.value(expr, caller_inline_info, iinfo)
            else:
                assert 'unreachable'

        if do_inline:
            arg_typs = template._inline_overloads[sig.args]['folded_args']
            # pass is typed so use the callee globals
            inline_closure_call(state.func_ir, impl.__globals__,
                                block, i, impl, typingctx=state.typingctx,
                                arg_typs=arg_typs,
                                typemap=state.type_annotation.typemap,
                                calltypes=state.type_annotation.calltypes,
                                work_list=work_list,
                                replace_freevars=False,
                                callee_validator=callee_ir_validator)
            return True
        else:
            return False


@register_pass(mutates_CFG=False, analysis_only=False)
class DeadCodeElimination(FunctionPass):
    """
    Does dead code elimination
    """

    _name = "dead_code_elimination"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        dead_code_elimination(state.func_ir, state.typemap)
        return True
