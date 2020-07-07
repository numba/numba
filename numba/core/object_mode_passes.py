from contextlib import contextmanager
import warnings
from numba.core import (errors, types, typing, funcdesc, config, pylowering,
                        transforms)
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
                                           register_pass)
from collections import defaultdict


@contextmanager
def giveup_context(state, msg):
    """
    Wraps code that would signal a fallback to interpreter mode
    """
    try:
        yield
    except Exception as e:
        if not state.status.can_giveup:
            raise
        else:
            # Clear all references attached to the traceback
            e = e.with_traceback(None)
            warnings.warn_explicit('%s: %s' % (msg, e),
                                   errors.NumbaWarning,
                                   state.func_id.filename,
                                   state.func_id.firstlineno)

            raise


@register_pass(mutates_CFG=True, analysis_only=False)
class ObjectModeFrontEnd(FunctionPass):
    _name = "object_mode_front_end"

    def __init__(self):
        FunctionPass.__init__(self)

    def _frontend_looplift(self, state):
        """
        Loop lifting analysis and transformation
        """
        loop_flags = state.flags.copy()
        outer_flags = state.flags.copy()
        # Do not recursively loop lift
        outer_flags.unset('enable_looplift')
        loop_flags.unset('enable_looplift')
        if not state.flags.enable_pyobject_looplift:
            loop_flags.unset('enable_pyobject')
        loop_flags.unset('enable_ssa')

        main, loops = transforms.loop_lifting(state.func_ir,
                                              typingctx=state.typingctx,
                                              targetctx=state.targetctx,
                                              locals=state.locals,
                                              flags=loop_flags)
        if loops:
            # Some loops were extracted
            if config.DEBUG_FRONTEND or config.DEBUG:
                for loop in loops:
                    print("Lifting loop", loop.get_source_location())
            from numba.core.compiler import compile_ir
            cres = compile_ir(state.typingctx, state.targetctx, main,
                              state.args, state.return_type,
                              outer_flags, state.locals,
                              lifted=tuple(loops), lifted_from=None,
                              is_lifted_loop=True)
            return cres

    def run_pass(self, state):
        from numba.core.compiler import _EarlyPipelineCompletion
        # NOTE: That so much stuff, including going back into the compiler, is
        # captured in a single pass is not ideal.
        if state.flags.enable_looplift:
            assert not state.lifted
            cres = self._frontend_looplift(state)
            if cres is not None:
                raise _EarlyPipelineCompletion(cres)

        # Fallback typing: everything is a python object
        state.typemap = defaultdict(lambda: types.pyobject)
        state.calltypes = defaultdict(lambda: types.pyobject)
        state.return_type = types.pyobject
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class ObjectModeBackEnd(LoweringPass):

    _name = "object_mode_back_end"

    def __init__(self):
        LoweringPass.__init__(self)

    def _py_lowering_stage(self, targetctx, library, interp, flags):
        fndesc = funcdesc.PythonFunctionDescriptor.from_object_mode_function(
            interp
        )
        with targetctx.push_code_library(library):
            lower = pylowering.PyLower(targetctx, library, fndesc, interp)
            lower.lower()
            if not flags.no_cpython_wrapper:
                lower.create_cpython_wrapper()
            env = lower.env
            call_helper = lower.call_helper
            del lower
        from numba.core.compiler import _LowerResult  # TODO: move this
        if flags.no_compile:
            return _LowerResult(fndesc, call_helper, cfunc=None, env=env)
        else:
            # Prepare for execution
            cfunc = targetctx.get_executable(library, fndesc, env)
            return _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)

    def run_pass(self, state):
        """
        Lowering for object mode
        """

        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        def backend_object_mode():
            """
            Object mode compilation
            """
            with giveup_context(state,
                                "Function %s failed at object mode lowering"
                                % (state.func_id.func_name,)):
                if len(state.args) != state.nargs:
                    # append missing
                    # BUG?: What's going on with nargs here?
                    # check state.nargs vs self.nargs on original code
                    state.args = (tuple(state.args) + (types.pyobject,) *
                                  (state.nargs - len(state.args)))

                return self._py_lowering_stage(state.targetctx,
                                               state.library,
                                               state.func_ir,
                                               state.flags)

        lowered = backend_object_mode()
        signature = typing.signature(state.return_type, *state.args)
        from numba.core.compiler import compile_result
        state.cr = compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            entry_point=lowered.cfunc,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            objectmode=True,
            interpmode=False,
            lifted=state.lifted,
            fndesc=lowered.fndesc,
            environment=lowered.env,
            metadata=state.metadata,
            reload_init=state.reload_init,
        )

        # Warn, deprecated behaviour, code compiled in objmode without
        # force_pyobject indicates fallback from nopython mode
        if not state.flags.force_pyobject:
            # first warn about object mode and yes/no to lifted loops
            if len(state.lifted) > 0:
                warn_msg = ('Function "%s" was compiled in object mode without'
                            ' forceobj=True, but has lifted loops.' %
                            (state.func_id.func_name,))
            else:
                warn_msg = ('Function "%s" was compiled in object mode without'
                            ' forceobj=True.' % (state.func_id.func_name,))
            warnings.warn(errors.NumbaWarning(warn_msg,
                                              state.func_ir.loc))

            url = ("http://numba.pydata.org/numba-doc/latest/reference/"
                   "deprecation.html#deprecation-of-object-mode-fall-"
                   "back-behaviour-when-using-jit")
            msg = ("\nFall-back from the nopython compilation path to the "
                   "object mode compilation path has been detected, this is "
                   "deprecated behaviour.\n\nFor more information visit %s" %
                   url)
            warnings.warn(errors.NumbaDeprecationWarning(msg,
                                                         state.func_ir.loc))
            if state.flags.release_gil:
                warn_msg = ("Code running in object mode won't allow parallel"
                            " execution despite nogil=True.")
                warnings.warn_explicit(warn_msg, errors.NumbaWarning,
                                       state.func_id.filename,
                                       state.func_id.firstlineno)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class CompileInterpMode(LoweringPass):

    _name = "compile_interp_mode"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Just create a compile result for interpreter mode
        """
        args = [types.pyobject] * len(state.args)
        signature = typing.signature(types.pyobject, *args)
        from numba.core.compiler import compile_result
        state.cr = compile_result(typing_context=state.typingctx,
                                  target_context=state.targetctx,
                                  entry_point=state.func_id.func,
                                  typing_error=state.status.fail_reason,
                                  type_annotation="<Interpreter mode function>",
                                  signature=signature,
                                  objectmode=False,
                                  interpmode=True,
                                  lifted=(),
                                  fndesc=None,)
        return True
