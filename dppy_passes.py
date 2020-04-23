from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import warnings

import weakref
from collections import namedtuple, deque
import operator

from numba.lowering import Lower, _VarArgItem

from llvmlite.llvmpy.core import Constant, Type, Builder
from numba.stencilparfor import StencilPass

from numba import (
    config,
    errors,
    types,
    rewrites,
    typeinfer,
    funcdesc,
    utils,
    typing,
    postproc,
    ir)


from numba.ir_utils import (
    dprint_func_ir,
    simplify_CFG,
    canonicalize_array_math,
    simplify,
    remove_dels,
    guard,
    dead_code_elimination
    )

from numba.errors import (LoweringError, new_error_context, TypingError,
                     LiteralTypingError)

from numba.compiler_machinery import FunctionPass, LoweringPass, register_pass

from .dppy_lowerer import DPPyLower

from numba.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfor import ParforPass as _parfor_ParforPass
from numba.parfor import Parfor
#from numba.npyufunc.dufunc import DUFunc


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyPreParforPass(FunctionPass):

    _name = "dppy_pre_parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Preprocessing for data-parallel computations.
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        print(state.flags.auto_parallel.numpy)

        preparfor_pass = _parfor_PreParforPass(
            state.func_ir,
            state.type_annotation.typemap,
            state.type_annotation.calltypes, state.typingctx,
            state.flags.auto_parallel,
            state.parfor_diagnostics.replaced_fns
        )

        preparfor_pass.run()

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()

        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyParforPass(FunctionPass):

    _name = "dppy_parfor_pass"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Convert data-parallel computations into Parfor nodes
        """
        # Ensure we have an IR and type information.
        assert state.func_ir
        parfor_pass = _parfor_ParforPass(state.func_ir,
                                         state.type_annotation.typemap,
                                         state.type_annotation.calltypes,
                                         state.return_type,
                                         state.typingctx,
                                         state.flags.auto_parallel,
                                         state.flags,
                                         state.parfor_diagnostics)

        parfor_pass.run()

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()

        return True


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


@register_pass(mutates_CFG=True, analysis_only=False)
class SpirvFriendlyLowering(LoweringPass):

    _name = "spirv_friendly_lowering"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            # Enable object caching upfront, so that the library can
            # be later serialized.
            state.library.enable_object_caching()

        targetctx = state.targetctx
        library   = state.library
        interp    = state.func_ir  # why is it called this?!
        typemap   = state.typemap
        restype   = state.return_type
        calltypes = state.calltypes
        flags     = state.flags
        metadata  = state.metadata

        msg = ("Function %s failed at nopython "
               "mode lowering" % (state.func_id.func_name,))
        with fallback_context(state, msg):
            # Lowering
            fndesc = \
                funcdesc.PythonFunctionDescriptor.from_specialized_function(
                    interp, typemap, restype, calltypes,
                    mangler=targetctx.mangler, inline=flags.forceinline,
                    noalias=flags.noalias)

            with targetctx.push_code_library(library):
                lower = DPPyLower(targetctx, library, fndesc, interp,
                                       metadata=metadata)
                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)
                env = lower.env
                call_helper = lower.call_helper
                del lower

            from numba.compiler import _LowerResult  # TODO: move this
            if flags.no_compile:
                state['cr'] = _LowerResult(fndesc, call_helper,
                                           cfunc=None, env=env)
            else:
                # Prepare for execution
                cfunc = targetctx.get_executable(library, fndesc, env)
                # Insert native function for use by other jitted-functions.
                # We also register its library to allow for inlining.
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state['cr'] = _LowerResult(fndesc, call_helper,
                                           cfunc=cfunc, env=env)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DPPyNoPythonBackend(FunctionPass):

    _name = "nopython_backend"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
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
class InlineParforVectorize(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.vectorize
    decorator directly into the site of its call depending on the value set in
    the 'inline' kwarg to the decorator.

    This is a typed pass. CFG simplification and DCE are performed on
    completion.
    """

    _name = "inline_parfor_vectorize"

    def __init__(self):
        FunctionPass.__init__(self)

    _DEBUG = True

    def run_pass(self, state):
        """Run inlining of overloads
        """
        if self._DEBUG:
            print('before vectorize inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))


        modified = False
        work_list = list(state.func_ir.blocks.items())
        # use a work list, look for call sites via `ir.Expr.op == call` and
        # then pass these to `self._do_work` to make decisions about inlining.
        while work_list:
            label, block = work_list.pop()
            for i, instr in enumerate(block.body):

                if isinstance(instr, Parfor):
                    # work through the loop body
                    for (l, b) in instr.loop_body.items():
                        for j, inst in enumerate(b.body):
                            if isinstance(inst, ir.Assign):
                                expr = inst.value
                                if isinstance(expr, ir.Expr):
                                    if expr.op == 'call':
                                        find_assn = b.find_variable_assignment(expr.func.name).value
                                        if isinstance(find_assn, ir.Global):
                                            # because of circular import, find better solution
                                            if (find_assn.value.__class__.__name__ == "DUFunc"):
                                                py_func = find_assn.value.py_func
                                                workfn = self._do_work_call(state, work_list,
                                                                            b, j, expr, py_func)

                                        print("Found call ", str(expr))
                                    else:
                                        continue

                                    #if guard(workfn, state, work_list, b, j, expr):
                                    if workfn:
                                        modified = True
                                        break  # because block structure changed


        if self._DEBUG:
            print('after vectorize inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))

        if modified:
            # clean up blocks
            dead_code_elimination(state.func_ir,
                                  typemap=state.type_annotation.typemap)
            # clean up unconditional branches that appear due to inlined
            # functions introducing blocks
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        if self._DEBUG:
            print('after vectorize inline DCE'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))

        return True

    def _do_work_call(self, state, work_list, block, i, expr, py_func):
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

        sig = state.type_annotation.calltypes[expr]
        is_method = False

        templates = getattr(func_ty, 'templates', None)
        arg_typs = sig.args

        if templates is None:
            return False

        # at this point we know we maybe want to inline something and there's
        # definitely something that could be inlined.
        return self._run_inliner(
            state, sig, templates[0], arg_typs, expr, i, py_func, block,
            work_list
        )

    def _run_inliner(
        self, state, sig, template, arg_typs, expr, i, impl, block,
        work_list
    ):
        from numba.inline_closurecall import (inline_closure_call,
                                              callee_ir_validator)

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

    def _add_method_self_arg(self, state, expr):
        func_def = guard(get_definition, state.func_ir, expr.func)
        if func_def is None:
            return False
        expr.args.insert(0, func_def.value)
        return True


