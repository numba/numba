from __future__ import print_function, division, absolute_import
from .compiler_machinery import FunctionPass, register_pass
from . import (config, bytecode, interpreter, postproc, errors, types, rewrites,
               transforms, ir, utils)
import warnings
from .analysis import (
    dead_branch_prune,
    rewrite_semantic_constants,
    find_literally_calls,
)
from contextlib import contextmanager
from .inline_closurecall import InlineClosureCallPass
from .ir_utils import (guard, resolve_func_from_module, simplify_CFG,
                       GuardException)


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
class ExtractByteCode(FunctionPass):
    _name = "extract_bytecode"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Extract bytecode from function
        """
        func_id = state['func_id']
        bc = bytecode.ByteCode(func_id)
        if config.DUMP_BYTECODE:
            print(bc.dump())

        state['bc'] = bc
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class TranslateByteCode(FunctionPass):
    _name = "translate_bytecode"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Analyze bytecode and translating to Numba IR
        """
        func_id = state['func_id']
        bc = state['bc']
        interp = interpreter.Interpreter(func_id)
        func_ir = interp.interpret(bc)
        state["func_ir"] = func_ir
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class FixupArgs(FunctionPass):
    _name = "fixup_args"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state['nargs'] = state['func_ir'].arg_count
        if not state['args'] and state['flags'].force_pyobject:
            # Allow an empty argument types specification when object mode
            # is explicitly requested.
            state['args'] = (types.pyobject,) * state['nargs']
        elif len(state['args']) != state['nargs']:
            raise TypeError("Signature mismatch: %d argument types given, "
                            "but function takes %d arguments"
                            % (len(state['args']), state['nargs']))
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class IRProcessing(FunctionPass):
    _name = "ir_processing"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_ir = state['func_ir']
        post_proc = postproc.PostProcessor(func_ir)
        post_proc.run()

        if config.DEBUG or config.DUMP_IR:
            name = func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            func_ir.dump()
            if func_ir.is_generator:
                print(("GENERATOR INFO: %s" % name).center(80, "-"))
                func_ir.dump_generator_info()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class RewriteSemanticConstants(FunctionPass):
    _name = "rewrite_semantic_constants"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
        assert state.func_ir
        msg = ('Internal error in pre-inference dead branch pruning '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            rewrite_semantic_constants(state.func_ir, state.args)

        if config.DEBUG or config.DUMP_IR:
            print('branch_pruned_ir'.center(80, '-'))
            print(state.func_ir.dump())
            print('end branch_pruned_ir'.center(80, '-'))
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class DeadBranchPrune(FunctionPass):
    _name = "dead_branch_prune"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """

        # purely for demonstration purposes, obtain the analysis from a pass
        # declare as a required dependent
        semantic_const_analysis = self.get_analysis(type(self))  # noqa

        assert state.func_ir
        msg = ('Internal error in pre-inference dead branch pruning '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            dead_branch_prune(state.func_ir, state.args)

        if config.DEBUG or config.DUMP_IR:
            print('branch_pruned_ir'.center(80, '-'))
            print(state.func_ir.dump())
            print('end branch_pruned_ir'.center(80, '-'))

        return True

    def get_analysis_usage(self, AU):
        AU.add_required(RewriteSemanticConstants)


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineClosureLikes(FunctionPass):
    _name = "inline_closure_likes"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # Ensure we have an IR and type information.
        assert state.func_ir

        # if the return type is a pyobject, there's no type info available and
        # no ability to resolve certain typed function calls in the array
        # inlining code, use this variable to indicate
        typed_pass = not isinstance(state.return_type, types.misc.PyObject)
        inline_pass = InlineClosureCallPass(state.func_ir,
                                            state.flags.auto_parallel,
                                            state.parfor_diagnostics.replaced_fns,
                                            typed_pass)
        inline_pass.run()
        # Remove all Dels, and re-run postproc
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run()

        if config.DEBUG or config.DUMP_IR:
            name = state.func_ir.func_id.func_qualname
            print(("IR DUMP: %s" % name).center(80, "-"))
            state.func_ir.dump()
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class GenericRewrites(FunctionPass):
    _name = "generic_rewrites"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Perform any intermediate representation rewrites before type
        inference.
        """
        assert state.func_ir
        msg = ('Internal error in pre-inference rewriting '
               'pass encountered during compilation of '
               'function "%s"' % (state.func_id.func_name,))
        with fallback_context(state, msg):
            rewrites.rewrite_registry.apply('before-inference', state)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class WithLifting(FunctionPass):
    _name = "with_lifting"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Extract with-contexts
        """
        main, withs = transforms.with_lifting(
            func_ir=state.func_ir,
            typingctx=state.typingctx,
            targetctx=state.targetctx,
            flags=state.flags,
            locals=state.locals,
        )
        if withs:
            from numba.compiler import compile_ir, _EarlyPipelineCompletion
            cres = compile_ir(state.typingctx, state.targetctx, main,
                              state.args, state.return_type,
                              state.flags, state.locals,
                              lifted=tuple(withs), lifted_from=None,
                              pipeline_class=type(state.pipeline))
            raise _EarlyPipelineCompletion(cres)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class InlineInlinables(FunctionPass):
    """
    This pass will inline a function wrapped by the numba.jit decorator directly
    into the site of its call depending on the value set in the 'inline' kwarg
    to the decorator.

    This is an untyped pass. CFG simplification is performed at the end of the
    pass but no block level clean up is performed on the mutated IR (typing
    information is not available to do so).
    """
    _name = "inline_inlinables"
    _DEBUG = False

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """Run inlining of inlinables
        """
        if config.DEBUG or self._DEBUG:
            print('before inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        modified = False
        # use a work list, look for call sites via `ir.Expr.op == call` and
        # then pass these to `self._do_work` to make decisions about inlining.
        work_list = list(state.func_ir.blocks.items())
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

        if modified:
            # clean up unconditional branches that appear due to inlined
            # functions introducing blocks
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)

        if config.DEBUG or self._DEBUG:
            print('after inline'.center(80, '-'))
            print(state.func_ir.dump())
            print(''.center(80, '-'))
        return True

    def _do_work(self, state, work_list, block, i, expr):
        from numba.inline_closurecall import (inline_closure_call,
                                              callee_ir_validator)
        from numba.compiler import run_frontend
        from numba.targets.cpu import InlineOptions

        # try and get a definition for the call, this isn't always possible as
        # it might be a eval(str)/part generated awaiting update etc. (parfors)
        to_inline = None
        try:
            to_inline = state.func_ir.get_definition(expr.func)
        except Exception:
            if self._DEBUG:
                print("Cannot find definition for %s" % expr.func)
            return False
        # do not handle closure inlining here, another pass deals with that.
        if getattr(to_inline, 'op', False) == 'make_function':
            return False

        # see if the definition is a "getattr", in which case walk the IR to
        # try and find the python function via the module from which it's
        # imported, this should all be encoded in the IR.
        if getattr(to_inline, 'op', False) == 'getattr':
            val = resolve_func_from_module(state.func_ir, to_inline)
        else:
            # This is likely a freevar or global
            #
            # NOTE: getattr 'value' on a call may fail if it's an ir.Expr as
            # getattr is overloaded to look in _kws.
            try:
                val = getattr(to_inline, 'value', False)
            except Exception:
                raise GuardException

        # if something was found...
        if val:
            # check it's dispatcher-like, the targetoptions attr holds the
            # kwargs supplied in the jit decorator and is where 'inline' will
            # be if it is present.
            topt = getattr(val, 'targetoptions', False)
            if topt:
                inline_type = topt.get('inline', None)
                # has 'inline' been specified?
                if inline_type is not None:
                    inline_opt = InlineOptions(inline_type)
                    # Could this be inlinable?
                    if not inline_opt.is_never_inline:
                        # yes, it could be inlinable
                        do_inline = True
                        pyfunc = val.py_func
                        # Has it got an associated cost model?
                        if inline_opt.has_cost_model:
                            # yes, it has a cost model, use it to determine
                            # whether to do the inline
                            py_func_ir = run_frontend(pyfunc)
                            do_inline = inline_type(expr, state.func_ir,
                                                    py_func_ir)
                        # if do_inline is True then inline!
                        if do_inline:
                            inline_closure_call(state.func_ir,
                                                pyfunc.__globals__,
                                                block, i, pyfunc,
                                                work_list=work_list,
                                                callee_validator=callee_ir_validator)
                            return True
        return False


@register_pass(mutates_CFG=False, analysis_only=False)
class PreserveIR(FunctionPass):
    """
    Preserves the IR in the metadata
    """

    _name = "preserve_ir"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.metadata['preserved_ir'] = state.func_ir.copy()
        return False


@register_pass(mutates_CFG=False, analysis_only=True)
class FindLiterallyCalls(FunctionPass):
    """Find calls to `numba.literally()` and signal if its requirement is not
    satisfied.
    """
    _name = "find_literally"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        find_literally_calls(state.func_ir, state.args)
        return False
