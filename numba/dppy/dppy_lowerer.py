from __future__ import print_function, division, absolute_import
from contextlib import contextmanager
import warnings

import weakref
from collections import namedtuple, deque
import operator

from numba.lowering import BaseLower, _VarArgItem

from llvmlite.llvmpy.core import Constant, Type, Builder

from numba import (
    config,
    errors,
    types,
    rewrites,
    typeinfer,
    funcdesc,
    utils,
    typing,
    ir,
    ir_utils,
    cgutils,
    generators)

from numba.errors import (LoweringError, new_error_context, TypingError,
                     LiteralTypingError)

from numba.compiler_machinery import FunctionPass, LoweringPass, register_pass


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


class DPPyLower(BaseLower):
    GeneratorLower = generators.GeneratorLower

    def pre_block(self, block):
        from numba.unsafe import eh

        super(DPPyLower, self).pre_block(block)

        # Detect if we are in a TRY block by looking for a call to
        # `eh.exception_check`.
        for call in block.find_exprs(op='call'):
            defn = ir_utils.guard(
                ir_utils.get_definition, self.func_ir, call.func,
            )
            if defn is not None and isinstance(defn, ir.Global):
                if defn.value is eh.exception_check:
                    if isinstance(block.terminator, ir.Branch):
                        targetblk = self.blkmap[block.terminator.truebr]
                        # NOTE: This hacks in an attribute for call_conv to
                        #       pick up. This hack is no longer needed when
                        #       all old-style implementations are gone.
                        self.builder._in_try_block = {'target': targetblk}
                        break

    def post_block(self, block):
        # Clean-up
        try:
            del self.builder._in_try_block
        except AttributeError:
            pass

    def lower_inst(self, inst):
        # Set debug location for all subsequent LL instructions
        self.debuginfo.mark_location(self.builder, self.loc)
        self.debug_print(str(inst))
        if isinstance(inst, ir.Assign):
            ty = self.typeof(inst.target.name)
            val = self.lower_assign(ty, inst)
            self.storevar(val, inst.target.name)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]

            condty = self.typeof(inst.cond.name)
            pred = self.context.cast(self.builder, cond, condty, types.boolean)
            assert pred.type == Type.int(1), ("cond is not i1: %s" % pred.type)
            self.builder.cbranch(pred, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Return):
            if self.generator_info:
                # StopIteration
                self.genlower.return_from_generator(self)
                return
            val = self.loadvar(inst.value.name)
            oty = self.typeof(inst.value.name)
            ty = self.fndesc.restype
            if isinstance(ty, types.Optional):
                # If returning an optional type
                self.call_conv.return_optional_value(self.builder, ty, oty, val)
                return
            if ty != oty:
                val = self.context.cast(self.builder, val, oty, ty)
            retval = self.context.get_return_value(self.builder, ty, val)
            self.call_conv.return_value(self.builder, retval)

        elif isinstance(inst, ir.StaticSetItem):
            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            try:
                impl = self.context.get_function('static_setitem', signature)
            except NotImplementedError:
                return self.lower_setitem(inst.target, inst.index_var,
                                          inst.value, signature)
            else:
                target = self.loadvar(inst.target.name)
                value = self.loadvar(inst.value.name)
                valuety = self.typeof(inst.value.name)
                value = self.context.cast(self.builder, value, valuety,
                                          signature.args[2])
                return impl(self.builder, (target, inst.index, value))

        elif isinstance(inst, ir.Print):
            self.lower_print(inst)

        elif isinstance(inst, ir.SetItem):
            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            return self.lower_setitem(inst.target, inst.index, inst.value,
                                      signature)

        elif isinstance(inst, ir.StoreMap):
            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            return self.lower_setitem(inst.dct, inst.key, inst.value, signature)

        elif isinstance(inst, ir.DelItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)

            targetty = self.typeof(inst.target.name)
            indexty = self.typeof(inst.index.name)

            signature = self.fndesc.calltypes[inst]
            assert signature is not None

            op = operator.delitem
            fnop = self.context.typing_context.resolve_value_type(op)
            callsig = fnop.get_call_type(
                self.context.typing_context, signature.args, {},
            )
            impl = self.context.get_function(fnop, callsig)

            assert targetty == signature.args[0]
            index = self.context.cast(self.builder, index, indexty,
                                      signature.args[1])

            return impl(self.builder, (target, index))

        elif isinstance(inst, ir.Del):
            self.delvar(inst.value)

        elif isinstance(inst, ir.SetAttr):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            signature = self.fndesc.calltypes[inst]

            targetty = self.typeof(inst.target.name)
            valuety = self.typeof(inst.value.name)
            assert signature is not None
            assert signature.args[0] == targetty
            impl = self.context.get_setattr(inst.attr, signature)

            # Convert argument to match
            value = self.context.cast(self.builder, value, valuety,
                                      signature.args[1])

            return impl(self.builder, (target, value))

        elif isinstance(inst, ir.StaticRaise):
            self.lower_static_raise(inst)

        elif isinstance(inst, ir.StaticTryRaise):
            self.lower_static_try_raise(inst)

        else:
            print(type(inst))
            for _class, func in lower_extensions.items():
                if isinstance(inst, _class):
                    func(self, inst)
                    return
            raise NotImplementedError(type(inst))

    def lower_setitem(self, target_var, index_var, value_var, signature):
        target = self.loadvar(target_var.name)
        value = self.loadvar(value_var.name)
        index = self.loadvar(index_var.name)

        targetty = self.typeof(target_var.name)
        valuety = self.typeof(value_var.name)
        indexty = self.typeof(index_var.name)

        op = operator.setitem
        fnop = self.context.typing_context.resolve_value_type(op)
        callsig = fnop.get_call_type(
            self.context.typing_context, signature.args, {},
        )
        impl = self.context.get_function(fnop, callsig)

        # Convert argument to match
        if isinstance(targetty, types.Optional):
            target = self.context.cast(self.builder, target, targetty,
                                       targetty.type)
        else:
            assert targetty == signature.args[0]

        index = self.context.cast(self.builder, index, indexty,
                                  signature.args[1])
        value = self.context.cast(self.builder, value, valuety,
                                  signature.args[2])

        return impl(self.builder, (target, index, value))

    def lower_static_raise(self, inst):
        if inst.exc_class is None:
            # Reraise
            self.return_exception(None, loc=self.loc)
        else:
            self.return_exception(inst.exc_class, inst.exc_args, loc=self.loc)

    def lower_static_try_raise(self, inst):
        if inst.exc_class is None:
            # Reraise
            self.set_exception(None, loc=self.loc)
        else:
            self.set_exception(inst.exc_class, inst.exc_args, loc=self.loc)

    def lower_assign(self, ty, inst):
        value = inst.value
        # In nopython mode, closure vars are frozen like globals
        if isinstance(value, (ir.Const, ir.Global, ir.FreeVar)):
            res = self.context.get_constant_generic(self.builder, ty,
                                                    value.value)
            self.incref(ty, res)
            return res

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            res = self.context.cast(self.builder, val, oty, ty)
            self.incref(ty, res)
            return res

        elif isinstance(value, ir.Arg):
            # Cast from the argument type to the local variable type
            # (note the "arg.FOO" convention as used in typeinfer)
            argty = self.typeof("arg." + value.name)
            if isinstance(argty, types.Omitted):
                pyval = argty.value
                # use the type of the constant value
                valty = self.context.typing_context.resolve_value_type(pyval)
                const = self.context.get_constant_generic(self.builder, valty,
                                                          pyval)
                # cast it to the variable type
                res = self.context.cast(self.builder, const, valty, ty)
            else:
                val = self.fnargs[value.index]
                res = self.context.cast(self.builder, val, argty, ty)
            self.incref(ty, res)
            return res

        elif isinstance(value, ir.Yield):
            res = self.lower_yield(ty, value)
            self.incref(ty, res)
            return res

        raise NotImplementedError(type(value), value)

    def lower_yield(self, retty, inst):
        yp = self.generator_info.yield_points[inst.index]
        assert yp.inst is inst
        y = generators.LowerYield(self, yp, yp.live_vars)
        y.lower_yield_suspend()
        # Yield to caller
        val = self.loadvar(inst.value.name)
        typ = self.typeof(inst.value.name)

        # cast the local val to the type yielded
        yret = self.context.cast(self.builder, val, typ,
                                 self.gentype.yield_type)

        # get the return repr of yielded value
        retval = self.context.get_return_value(self.builder, typ, yret)

        # return
        self.call_conv.return_value(self.builder, retval)

        # Resumption point
        y.lower_yield_resume()
        # None is returned by the yield expression
        return self.context.get_constant_generic(self.builder, retty, None)

    def lower_binop(self, resty, expr, op):
        # if op in utils.OPERATORS_TO_BUILTINS:
        # map operator.the_op => the corresponding types.Function()
        # TODO: is this looks dodgy ...
        op = self.context.typing_context.resolve_value_type(op)

        lhs = expr.lhs
        rhs = expr.rhs
        static_lhs = expr.static_lhs
        static_rhs = expr.static_rhs
        lty = self.typeof(lhs.name)
        rty = self.typeof(rhs.name)
        lhs = self.loadvar(lhs.name)
        rhs = self.loadvar(rhs.name)

        # Convert argument to match
        signature = self.fndesc.calltypes[expr]
        lhs = self.context.cast(self.builder, lhs, lty, signature.args[0])
        rhs = self.context.cast(self.builder, rhs, rty, signature.args[1])

        def cast_result(res):
            return self.context.cast(self.builder, res,
                                     signature.return_type, resty)

        # First try with static operands, if known
        def try_static_impl(tys, args):
            if any(a is ir.UNDEFINED for a in args):
                return None
            try:
                if isinstance(op, types.Function):
                    static_sig = op.get_call_type(self.context.typing_context,
                                                  tys, {})
                else:
                    static_sig = typing.signature(signature.return_type, *tys)
            except TypingError:
                return None
            try:
                static_impl = self.context.get_function(op, static_sig)
                return static_impl(self.builder, args)
            except NotImplementedError:
                return None

        res = try_static_impl(
            (_lit_or_omitted(static_lhs), _lit_or_omitted(static_rhs)),
            (static_lhs, static_rhs),
        )
        if res is not None:
            return cast_result(res)

        res = try_static_impl(
            (_lit_or_omitted(static_lhs), rty),
            (static_lhs, rhs),
        )
        if res is not None:
            return cast_result(res)

        res = try_static_impl(
            (lty, _lit_or_omitted(static_rhs)),
            (lhs, static_rhs),
        )
        if res is not None:
            return cast_result(res)

        # Normal implementation for generic arguments

        sig = op.get_call_type(self.context.typing_context, signature.args, {})
        impl = self.context.get_function(op, sig)
        res = impl(self.builder, (lhs, rhs))
        return cast_result(res)

    def lower_getitem(self, resty, expr, value, index, signature):
        baseval = self.loadvar(value.name)
        indexval = self.loadvar(index.name)
        # Get implementation of getitem
        op = operator.getitem
        fnop = self.context.typing_context.resolve_value_type(op)
        callsig = fnop.get_call_type(
            self.context.typing_context, signature.args, {},
        )
        impl = self.context.get_function(fnop, callsig)

        argvals = (baseval, indexval)
        argtyps = (self.typeof(value.name),
                   self.typeof(index.name))
        castvals = [self.context.cast(self.builder, av, at, ft)
                    for av, at, ft in zip(argvals, argtyps,
                                          signature.args)]
        res = impl(self.builder, castvals)
        return self.context.cast(self.builder, res,
                                 signature.return_type,
                                 resty)

    def _cast_var(self, var, ty):
        """
        Cast a Numba IR variable to the given Numba type, returning a
        low-level value.
        """
        if isinstance(var, _VarArgItem):
            varty = self.typeof(var.vararg.name)[var.index]
            val = self.builder.extract_value(self.loadvar(var.vararg.name),
                                             var.index)
        else:
            varty = self.typeof(var.name)
            val = self.loadvar(var.name)
        return self.context.cast(self.builder, val, varty, ty)

    def fold_call_args(self, fnty, signature, pos_args, vararg, kw_args):
        if vararg:
            # Inject *args from function call
            # The lowering will be done in _cast_var() above.
            tp_vararg = self.typeof(vararg.name)
            assert isinstance(tp_vararg, types.BaseTuple)
            pos_args = pos_args + [_VarArgItem(vararg, i)
                                   for i in range(len(tp_vararg))]

        # Fold keyword arguments and resolve default argument values
        pysig = signature.pysig
        if pysig is None:
            if kw_args:
                raise NotImplementedError("unsupported keyword arguments "
                                          "when calling %s" % (fnty,))
            argvals = [self._cast_var(var, sigty)
                       for var, sigty in zip(pos_args, signature.args)]
        else:
            def normal_handler(index, param, var):
                return self._cast_var(var, signature.args[index])

            def default_handler(index, param, default):
                return self.context.get_constant_generic(
                    self.builder, signature.args[index], default)

            def stararg_handler(index, param, vars):
                stararg_ty = signature.args[index]
                assert isinstance(stararg_ty, types.BaseTuple), stararg_ty
                values = [self._cast_var(var, sigty)
                          for var, sigty in zip(vars, stararg_ty)]
                return cgutils.make_anonymous_struct(self.builder, values)

            argvals = typing.fold_arguments(pysig,
                                            pos_args, dict(kw_args),
                                            normal_handler,
                                            default_handler,
                                            stararg_handler)
        return argvals

    def lower_print(self, inst):
        """
        Lower a ir.Print()
        """
        # We handle this, as far as possible, as a normal call to built-in
        # print().  This will make it easy to undo the special ir.Print
        # rewrite when it becomes unnecessary (e.g. when we have native
        # strings).
        sig = self.fndesc.calltypes[inst]
        assert sig.return_type == types.none
        fnty = self.context.typing_context.resolve_value_type(print)

        # Fix the call signature to inject any constant-inferred
        # string argument
        pos_tys = list(sig.args)
        pos_args = list(inst.args)
        for i in range(len(pos_args)):
            if i in inst.consts:
                pyval = inst.consts[i]
                if isinstance(pyval, str):
                    pos_tys[i] = types.literal(pyval)

        fixed_sig = typing.signature(sig.return_type, *pos_tys)
        fixed_sig.pysig = sig.pysig

        argvals = self.fold_call_args(fnty, sig, pos_args, inst.vararg, {})
        impl = self.context.get_function(print, fixed_sig)
        impl(self.builder, argvals)

    def lower_call(self, resty, expr):
        signature = self.fndesc.calltypes[expr]
        self.debug_print("# lower_call: expr = {0}".format(expr))
        if isinstance(signature.return_type, types.Phantom):
            return self.context.get_dummy_value()

        if isinstance(expr.func, ir.Intrinsic):
            fnty = expr.func.name
        else:
            fnty = self.typeof(expr.func.name)

        if isinstance(fnty, types.ObjModeDispatcher):
            res = self._lower_call_ObjModeDispatcher(fnty, expr, signature)

        elif isinstance(fnty, types.ExternalFunction):
            res = self._lower_call_ExternalFunction(fnty, expr, signature)

        elif isinstance(fnty, types.ExternalFunctionPointer):
            res = self._lower_call_ExternalFunctionPointer(
                fnty, expr, signature)

        elif isinstance(fnty, types.RecursiveCall):
            res = self._lower_call_RecursiveCall(fnty, expr, signature)

        else:
            res = self._lower_call_normal(fnty, expr, signature)

        # If lowering the call returned None, interpret that as returning dummy
        # value if the return type of the function is void, otherwise there is
        # a problem
        if res is None:
            if signature.return_type == types.void:
                res = self.context.get_dummy_value()
            else:
                raise LoweringError(
                    msg="non-void function returns None from implementation",
                    loc=self.loc
                )

        return self.context.cast(self.builder, res, signature.return_type,
                                 resty)

    def _lower_call_ObjModeDispatcher(self, fnty, expr, signature):
        self.init_pyapi()
        # Acquire the GIL
        gil_state = self.pyapi.gil_ensure()
        # Fix types
        argnames = [a.name for a in expr.args]
        argtypes = [self.typeof(a) for a in argnames]
        argvalues = [self.loadvar(a) for a in argnames]
        for v, ty in zip(argvalues, argtypes):
            # Because .from_native_value steal the reference
            self.incref(ty, v)

        argobjs = [self.pyapi.from_native_value(atyp, aval,
                                                self.env_manager)
                   for atyp, aval in zip(argtypes, argvalues)]
        # Make Call
        entry_pt = fnty.dispatcher.compile(tuple(argtypes))
        callee = self.context.add_dynamic_addr(
            self.builder,
            id(entry_pt),
            info="with_objectmode",
        )
        ret_obj = self.pyapi.call_function_objargs(callee, argobjs)
        has_exception = cgutils.is_null(self.builder, ret_obj)
        with self. builder.if_else(has_exception) as (then, orelse):
            # Handles exception
            # This branch must exit the function
            with then:
                # Clean arg
                for obj in argobjs:
                    self.pyapi.decref(obj)

                # Release the GIL
                self.pyapi.gil_release(gil_state)

                # Return and signal exception
                self.call_conv.return_exc(self.builder)

            # Handles normal return
            with orelse:
                # Fix output value
                native = self.pyapi.to_native_value(
                    fnty.dispatcher.output_types,
                    ret_obj,
                )
                output = native.value

                # Release objs
                self.pyapi.decref(ret_obj)
                for obj in argobjs:
                    self.pyapi.decref(obj)

                # cleanup output
                if callable(native.cleanup):
                    native.cleanup()

                # Release the GIL
                self.pyapi.gil_release(gil_state)

                # Error during unboxing
                with self.builder.if_then(native.is_error):
                    self.call_conv.return_exc(self.builder)

                return output

    def _lower_call_ExternalFunction(self, fnty, expr, signature):
        # Handle a named external function
        self.debug_print("# external function")
        argvals = self.fold_call_args(
            fnty, signature, expr.args, expr.vararg, expr.kws,
        )
        fndesc = funcdesc.ExternalFunctionDescriptor(
            fnty.symbol, fnty.sig.return_type, fnty.sig.args)
        func = self.context.declare_external_function(
            self.builder.module, fndesc)
        return self.context.call_external_function(
            self.builder, func, fndesc.argtypes, argvals,
        )

    def _lower_call_ExternalFunctionPointer(self, fnty, expr, signature):
        # Handle a C function pointer
        self.debug_print("# calling external function pointer")
        argvals = self.fold_call_args(
            fnty, signature, expr.args, expr.vararg, expr.kws,
        )
        pointer = self.loadvar(expr.func.name)
        # If the external function pointer uses libpython
        if fnty.requires_gil:
            self.init_pyapi()
            # Acquire the GIL
            gil_state = self.pyapi.gil_ensure()
            # Make PyObjects
            newargvals = []
            pyvals = []
            for exptyp, gottyp, aval in zip(fnty.sig.args, signature.args,
                                            argvals):
                # Adjust argument values to pyobjects
                if exptyp == types.ffi_forced_object:
                    self.incref(gottyp, aval)
                    obj = self.pyapi.from_native_value(
                        gottyp, aval, self.env_manager,
                    )
                    newargvals.append(obj)
                    pyvals.append(obj)
                else:
                    newargvals.append(aval)

            # Call external function
            res = self.context.call_function_pointer(
                self.builder, pointer, newargvals, fnty.cconv,
            )
            # Release PyObjects
            for obj in pyvals:
                self.pyapi.decref(obj)

            # Release the GIL
            self.pyapi.gil_release(gil_state)
        # If the external function pointer does NOT use libpython
        else:
            res = self.context.call_function_pointer(
                self.builder, pointer, argvals, fnty.cconv,
            )
        return res

    def _lower_call_RecursiveCall(self, fnty, expr, signature):
        # Recursive call
        argvals = self.fold_call_args(
            fnty, signature, expr.args, expr.vararg, expr.kws,
        )
        qualprefix = fnty.overloads[signature.args]
        mangler = self.context.mangler or default_mangler
        mangled_name = mangler(qualprefix, signature.args)
        # special case self recursion
        if self.builder.function.name.startswith(mangled_name):
            res = self.context.call_internal(
                self.builder, self.fndesc, signature, argvals,
            )
        else:
            res = self.context.call_unresolved(
                self.builder, mangled_name, signature, argvals,
            )
        return res

    def _lower_call_normal(self, fnty, expr, signature):
        # Normal function resolution
        self.debug_print("# calling normal function: {0}".format(fnty))
        self.debug_print("# signature: {0}".format(signature))
        if (isinstance(expr.func, ir.Intrinsic) or
                isinstance(fnty, types.ObjModeDispatcher)):
            argvals = expr.func.args
        else:
            argvals = self.fold_call_args(
                fnty, signature, expr.args, expr.vararg, expr.kws,
            )
        impl = self.context.get_function(fnty, signature)
        if signature.recvr:
            # The "self" object is passed as the function object
            # for bounded function
            the_self = self.loadvar(expr.func.name)
            # Prepend the self reference
            argvals = [the_self] + list(argvals)

        res = impl(self.builder, argvals, self.loc)
        return res

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
            return self.lower_binop(resty, expr, expr.fn)
        elif expr.op == 'inplace_binop':
            lty = self.typeof(expr.lhs.name)
            if lty.mutable:
                return self.lower_binop(resty, expr, expr.fn)
            else:
                # inplace operators on non-mutable types reuse the same
                # definition as the corresponding copying operators.)
                return self.lower_binop(resty, expr, expr.immutable_fn)
        elif expr.op == 'unary':
            val = self.loadvar(expr.value.name)
            typ = self.typeof(expr.value.name)
            func_ty = self.context.typing_context.resolve_value_type(expr.fn)
            # Get function
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(func_ty, signature)
            # Convert argument to match
            val = self.context.cast(self.builder, val, typ, signature.args[0])
            res = impl(self.builder, [val])
            res = self.context.cast(self.builder, res,
                                    signature.return_type, resty)
            return res

        elif expr.op == 'call':
            res = self.lower_call(resty, expr)
            return res

        elif expr.op == 'pair_first':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            res = self.context.pair_first(self.builder, val, ty)
            self.incref(resty, res)
            return res

        elif expr.op == 'pair_second':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            res = self.context.pair_second(self.builder, val, ty)
            self.incref(resty, res)
            return res

        elif expr.op in ('getiter', 'iternext'):
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.op, signature)
            [fty] = signature.args
            castval = self.context.cast(self.builder, val, ty, fty)
            res = impl(self.builder, (castval,))
            res = self.context.cast(self.builder, res, signature.return_type,
                                    resty)
            return res

        elif expr.op == 'exhaust_iter':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            # Unpack optional
            if isinstance(ty, types.Optional):
                val = self.context.cast(self.builder, val, ty, ty.type)
                ty = ty.type

            # If we have a tuple, we needn't do anything
            # (and we can't iterate over the heterogeneous ones).
            if isinstance(ty, types.BaseTuple):
                assert ty == resty
                self.incref(ty, val)
                return val

            itemty = ty.iterator_type.yield_type
            tup = self.context.get_constant_undef(resty)
            pairty = types.Pair(itemty, types.boolean)
            getiter_sig = typing.signature(ty.iterator_type, ty)
            getiter_impl = self.context.get_function('getiter',
                                                     getiter_sig)
            iternext_sig = typing.signature(pairty, ty.iterator_type)
            iternext_impl = self.context.get_function('iternext',
                                                      iternext_sig)
            iterobj = getiter_impl(self.builder, (val,))
            # We call iternext() as many times as desired (`expr.count`).
            for i in range(expr.count):
                pair = iternext_impl(self.builder, (iterobj,))
                is_valid = self.context.pair_second(self.builder,
                                                    pair, pairty)
                with cgutils.if_unlikely(self.builder,
                                         self.builder.not_(is_valid)):
                    self.return_exception(ValueError, loc=self.loc)
                item = self.context.pair_first(self.builder,
                                               pair, pairty)
                tup = self.builder.insert_value(tup, item, i)

            # Call iternext() once more to check that the iterator
            # is exhausted.
            pair = iternext_impl(self.builder, (iterobj,))
            is_valid = self.context.pair_second(self.builder,
                                                pair, pairty)
            with cgutils.if_unlikely(self.builder, is_valid):
                self.return_exception(ValueError, loc=self.loc)

            self.decref(ty.iterator_type, iterobj)
            return tup

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)

            if isinstance(resty, types.BoundFunction):
                # if we are getting out a method, assume we have typed this
                # properly and just build a bound function object
                casted = self.context.cast(self.builder, val, ty, resty.this)
                res = self.context.get_bound_function(self.builder, casted,
                                                      resty.this)
                self.incref(resty, res)
                return res
            else:
                impl = self.context.get_getattr(ty, expr.attr)
                attrty = self.context.typing_context.resolve_getattr(ty,
                                                                     expr.attr)

                if impl is None:
                    # ignore the attribute
                    return self.context.get_dummy_value()
                else:
                    res = impl(self.context, self.builder, ty, val, expr.attr)

                # Cast the attribute type to the expected output type
                res = self.context.cast(self.builder, res, attrty, resty)
                return res

        elif expr.op == "static_getitem":
            signature = typing.signature(
                resty,
                self.typeof(expr.value.name),
                _lit_or_omitted(expr.index),
            )
            try:
                # Both get_function() and the returned implementation can
                # raise NotImplementedError if the types aren't supported
                impl = self.context.get_function("static_getitem", signature)
                return impl(self.builder,
                            (self.loadvar(expr.value.name), expr.index))
            except NotImplementedError:
                if expr.index_var is None:
                    raise
                # Fall back on the generic getitem() implementation
                # for this type.
                signature = self.fndesc.calltypes[expr]
                return self.lower_getitem(resty, expr, expr.value,
                                          expr.index_var, signature)
        elif expr.op == "typed_getitem":
            signature = typing.signature(
                resty,
                self.typeof(expr.value.name),
                self.typeof(expr.index.name),
            )
            impl = self.context.get_function("typed_getitem", signature)
            return impl(self.builder, (self.loadvar(expr.value.name),
                        self.loadvar(expr.index.name)))
        elif expr.op == "getitem":
            signature = self.fndesc.calltypes[expr]
            return self.lower_getitem(resty, expr, expr.value, expr.index,
                                      signature)

        elif expr.op == "build_tuple":
            itemvals = [self.loadvar(i.name) for i in expr.items]
            itemtys = [self.typeof(i.name) for i in expr.items]
            castvals = [self.context.cast(self.builder, val, fromty, toty)
                        for val, toty, fromty in zip(itemvals, resty, itemtys)]
            tup = self.context.make_tuple(self.builder, resty, castvals)
            self.incref(resty, tup)
            return tup

        elif expr.op == "build_list":
            itemvals = [self.loadvar(i.name) for i in expr.items]
            itemtys = [self.typeof(i.name) for i in expr.items]
            castvals = [self.context.cast(self.builder, val, fromty,
                                          resty.dtype)
                        for val, fromty in zip(itemvals, itemtys)]
            return self.context.build_list(self.builder, resty, castvals)

        elif expr.op == "build_set":
            # Insert in reverse order, as Python does
            items = expr.items[::-1]
            itemvals = [self.loadvar(i.name) for i in items]
            itemtys = [self.typeof(i.name) for i in items]
            castvals = [self.context.cast(self.builder, val, fromty,
                                          resty.dtype)
                        for val, fromty in zip(itemvals, itemtys)]
            return self.context.build_set(self.builder, resty, castvals)

        elif expr.op == "build_map":
            items = expr.items
            keys, values = [], []
            key_types, value_types = [], []
            for k, v in items:
                key = self.loadvar(k.name)
                keytype = self.typeof(k.name)
                val = self.loadvar(v.name)
                valtype = self.typeof(v.name)
                keys.append(key)
                values.append(val)
                key_types.append(keytype)
                value_types.append(valtype)
            return self.context.build_map(self.builder, resty,
                                          list(zip(key_types, value_types)),
                                          list(zip(keys, values)))

        elif expr.op == "cast":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            castval = self.context.cast(self.builder, val, ty, resty)
            self.incref(resty, castval)
            return castval

        elif expr.op in self.context.special_ops:
            res = self.context.special_ops[expr.op](self, expr)
            return res

        raise NotImplementedError(expr)

    def _alloca_var(self, name, fetype):
        """
        Ensure the given variable has an allocated stack slot.
        """
        if name not in self.varmap:
            # If not already defined, allocate it
            llty = self.context.get_value_type(fetype)
            ptr = self.alloca_lltype(name, llty)
            # Remember the pointer
            self.varmap[name] = ptr

    def getvar(self, name):
        """
        Get a pointer to the given variable's slot.
        """
        return self.varmap[name]

    def loadvar(self, name):
        """
        Load the given variable's value.
        """
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        """
        Store the value into the given variable.
        """
        fetype = self.typeof(name)

        # Define if not already
        self._alloca_var(name, fetype)

        # Clean up existing value stored in the variable
        old = self.loadvar(name)
        self.decref(fetype, old)

        # Store variable
        ptr = self.getvar(name)
        if value.type != ptr.type.pointee:
            msg = ("Storing {value.type} to ptr of {ptr.type.pointee} "
                   "('{name}'). FE type {fetype}").format(value=value,
                                                          ptr=ptr,
                                                          fetype=fetype,
                                                          name=name)
            raise AssertionError(msg)

        self.builder.store(value, ptr)

    def delvar(self, name):
        """
        Delete the given variable.
        """
        fetype = self.typeof(name)

        # Define if not already (may happen if the variable is deleted
        # at the beginning of a loop, but only set later in the loop)
        self._alloca_var(name, fetype)

        ptr = self.getvar(name)
        self.decref(fetype, self.builder.load(ptr))
        # Zero-fill variable to avoid double frees on subsequent dels
        self.builder.store(Constant.null(ptr.type.pointee), ptr)

    def alloca(self, name, type):
        lltype = self.context.get_value_type(type)
        return self.alloca_lltype(name, lltype)

    def alloca_lltype(self, name, lltype):
        # Is user variable?
        is_uservar = not name.startswith('$')
        # Allocate space for variable
        aptr = cgutils.alloca_once(self.builder, lltype,
                                   name=name, zfill=False)
        if is_uservar:
            # Emit debug info for user variable
            sizeof = self.context.get_abi_sizeof(lltype)
            self.debuginfo.mark_variable(self.builder, aptr, name=name,
                                         lltype=lltype, size=sizeof,
                                         loc=self.loc)
        return aptr

    def incref(self, typ, val):
        if not self.context.enable_nrt:
            return

        self.context.nrt.incref(self.builder, typ, val)

    def decref(self, typ, val):
        if not self.context.enable_nrt:
            return

        self.context.nrt.decref(self.builder, typ, val)


def _lit_or_omitted(value):
    """Returns a Literal instance if the type of value is supported;
    otherwise, return `Omitted(value)`.
    """
    try:
        return types.literal(value)
    except LiteralTypingError:
        return types.Omitted(value)


def dppy_lower_array_expr(lowerer, expr):
    raise NotImplementedError(expr)
