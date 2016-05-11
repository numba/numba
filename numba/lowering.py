from __future__ import print_function, division, absolute_import

from collections import namedtuple
import sys

from llvmlite.ir import Value
from llvmlite.llvmpy.core import Constant, Type, Builder

from . import (_dynfunc, cgutils, config, funcdesc, generators, ir, types,
               typing, utils)
from .errors import LoweringError


class Environment(_dynfunc.Environment):
    __slots__ = ()

    @classmethod
    def from_fndesc(cls, fndesc):
        mod = fndesc.lookup_module()
        return cls(mod.__dict__)

    def __reduce__(self):
        return _rebuild_env, (self.globals['__name__'], self.consts)


def _rebuild_env(modname, consts):
    from . import serialize
    mod = serialize._rebuild_module(modname)
    env = Environment(mod.__dict__)
    env.consts[:] = consts
    return env


_VarArgItem = namedtuple("_VarArgItem", ("vararg", "index"))


class BaseLower(object):
    """
    Lower IR to LLVM
    """

    # If true, then can't cache LLVM module accross process calls
    has_dynamic_globals = False

    def __init__(self, context, library, fndesc, interp):
        self.context = context
        self.library = library
        self.fndesc = fndesc
        self.blocks = utils.SortedMap(utils.iteritems(interp.blocks))
        self.interp = interp
        self.call_conv = context.call_conv
        self.generator_info = self.interp.generator_info

        # Initialize LLVM
        self.module = self.library.create_ir_module(self.fndesc.unique_name)

        # Python execution environment (will be available to the compiled
        # function).
        self.env = Environment.from_fndesc(self.fndesc)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.firstblk = min(self.blocks.keys())
        self.loc = -1

        # Subclass initialization
        self.init()

    def init(self):
        pass

    def init_pyapi(self):
        """
        Init the Python API and Environment Manager for the function being
        lowered.
        """
        if self.pyapi is not None:
            return
        self.pyapi = self.context.get_python_api(self.builder)

        # Store environment argument for later use
        self.envarg = self.call_conv.get_env_argument(self.function)
        # Sanity check
        with cgutils.if_unlikely(self.builder,
                                 cgutils.is_null(self.builder, self.envarg)):
            self.pyapi.err_set_string(
                "PyExc_SystemError",
                "Numba internal error: object mode function called "
                "without an environment")
            self.call_conv.return_exc(self.builder)

        self.env_body = self.context.get_env_body(self.builder, self.envarg)
        self.pyapi.emit_environment_sentry(self.envarg)
        self.env_manager = self.pyapi.get_env_manager(self.env, self.env_body,
                                                      self.envarg)

    def pre_lower(self):
        """
        Called before lowering all blocks.
        """
        # A given Lower object can be used for several LL functions
        # (for generators) and it's important to use a new API and
        # EnvironmentManager.
        self.pyapi = None

    def post_lower(self):
        """
        Called after all blocks are lowered
        """

    def pre_block(self, block):
        """
        Called before lowering a block.
        """

    def return_exception(self, exc_class, exc_args=None):
        self.call_conv.return_user_exc(self.builder, exc_class, exc_args)

    def lower(self):
        if self.generator_info is None:
            self.genlower = None
            self.lower_normal_function(self.fndesc)
        else:
            self.genlower = self.GeneratorLower(self)
            self.gentype = self.genlower.gentype

            self.genlower.lower_init_func(self)
            self.genlower.lower_next_func(self)
            if self.gentype.has_finalizer:
                self.genlower.lower_finalize_func(self)

        if config.DUMP_LLVM:
            print(("LLVM DUMP %s" % self.fndesc).center(80, '-'))
            print(self.module)
            print('=' * 80)

        # Run target specific post lowering transformation
        self.context.post_lowering(self.module, self.library)

        # Materialize LLVM Module
        self.library.add_ir_module(self.module)

    def extract_function_arguments(self):
        self.fnargs = self.call_conv.decode_arguments(self.builder,
                                                      self.fndesc.argtypes,
                                                      self.function)
        return self.fnargs

    def lower_normal_function(self, fndesc):
        """
        Lower non-generator *fndesc*.
        """
        self.setup_function(fndesc)

        # Init argument values
        self.extract_function_arguments()
        entry_block_tail = self.lower_function_body()

        # Close tail of entry block
        self.builder.position_at_end(entry_block_tail)
        self.builder.branch(self.blkmap[self.firstblk])

    def lower_function_body(self):
        """
        Lower the current function's body, and return the entry block.
        """
        # Init Python blocks
        for offset in self.blocks:
            bname = "B%s" % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)

        self.pre_lower()
        # pre_lower() may have changed the current basic block
        entry_block_tail = self.builder.basic_block

        self.debug_print("# function begin: {0}".format(
            self.fndesc.unique_name))
        # Lower all blocks
        for offset, block in self.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()
        return entry_block_tail

    def lower_block(self, block):
        """
        Lower the given block.
        """
        self.pre_block(block)
        for inst in block.body:
            self.loc = inst.loc
            try:
                self.lower_inst(inst)
            except LoweringError:
                raise
            except Exception as e:
                msg = "Internal error:\n%s: %s" % (type(e).__name__, e)
                raise LoweringError(msg, inst.loc)

    def create_cpython_wrapper(self, release_gil=False):
        """
        Create CPython wrapper(s) around this function (or generator).
        """
        if self.genlower:
            self.context.create_cpython_wrapper(self.library,
                                                self.genlower.gendesc,
                                                self.env, self.call_helper,
                                                release_gil=release_gil)
        self.context.create_cpython_wrapper(self.library, self.fndesc,
                                            self.env, self.call_helper,
                                            release_gil=release_gil)

    def setup_function(self, fndesc):
        # Setup function
        self.function = self.context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder(self.entry_block)
        self.call_helper = self.call_conv.init_call_helper(self.builder)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]

    def debug_print(self, msg):
        if config.DEBUG_JIT:
            self.context.debug_print(self.builder, "DEBUGJIT: {0}".format(msg))


class Lower(BaseLower):
    GeneratorLower = generators.GeneratorLower

    def lower_inst(self, inst):
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
                return self.lower_setitem(inst.target, inst.index_var, inst.value, signature)
            else:
                target = self.loadvar(inst.target.name)
                value = self.loadvar(inst.value.name)
                valuety = self.typeof(inst.value.name)
                value = self.context.cast(self.builder, value, valuety,
                                          signature.args[2])
                return impl(self.builder, (target, inst.index, value))

        elif isinstance(inst, ir.SetItem):
            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            return self.lower_setitem(inst.target, inst.index, inst.value, signature)

        elif isinstance(inst, ir.DelItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)

            targetty = self.typeof(inst.target.name)
            indexty = self.typeof(inst.index.name)

            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            impl = self.context.get_function('delitem', signature)

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

        else:
            raise NotImplementedError(type(inst))

    def lower_setitem(self, target_var, index_var, value_var, signature):
        target = self.loadvar(target_var.name)
        value = self.loadvar(value_var.name)
        index = self.loadvar(index_var.name)

        targetty = self.typeof(target_var.name)
        valuety = self.typeof(value_var.name)
        indexty = self.typeof(index_var.name)

        impl = self.context.get_function('setitem', signature)

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
            self.return_exception(None)
        else:
            self.return_exception(inst.exc_class, inst.exc_args)

    def lower_assign(self, ty, inst):
        value = inst.value
        # In nopython mode, closure vars are frozen like globals
        if isinstance(value, (ir.Const, ir.Global, ir.FreeVar)):
            if isinstance(ty, types.ExternalFunctionPointer):
                self.has_dynamic_globals = True

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
        val = self.context.cast(self.builder, val, typ, self.gentype.yield_type)
        self.call_conv.return_value(self.builder, val)

        # Resumption point
        y.lower_yield_resume()
        # None is returned by the yield expression
        return self.context.get_constant_generic(self.builder, retty, None)

    def lower_binop(self, resty, expr, op):
        lhs = expr.lhs
        rhs = expr.rhs
        lty = self.typeof(lhs.name)
        rty = self.typeof(rhs.name)
        lhs = self.loadvar(lhs.name)
        rhs = self.loadvar(rhs.name)
        # Get function
        signature = self.fndesc.calltypes[expr]
        impl = self.context.get_function(op, signature)
        # Convert argument to match
        lhs = self.context.cast(self.builder, lhs, lty, signature.args[0])
        rhs = self.context.cast(self.builder, rhs, rty, signature.args[1])
        res = impl(self.builder, (lhs, rhs))
        return self.context.cast(self.builder, res,
                                 signature.return_type, resty)

    def lower_getitem(self, resty, expr, value, index, signature):
        baseval = self.loadvar(value.name)
        indexval = self.loadvar(index.name)
        impl = self.context.get_function("getitem", signature)
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

    def lower_call(self, resty, expr):
        signature = self.fndesc.calltypes[expr]
        if isinstance(signature.return_type, types.Phantom):
            return self.context.get_dummy_value()

        if isinstance(expr.func, ir.Intrinsic):
            fnty = expr.func.name
            argvals = expr.func.args
        else:
            fnty = self.typeof(expr.func.name)
            pos_args = expr.args
            if expr.vararg:
                # Inject *args from function call
                # The lowering will be done in _cast_var() above.
                tp_vararg = self.typeof(expr.vararg.name)
                assert isinstance(tp_vararg, types.BaseTuple)
                pos_args = pos_args + [_VarArgItem(expr.vararg, i)
                                       for i in range(len(tp_vararg))]

            # Fold keyword arguments and resolve default argument values
            pysig = signature.pysig
            if pysig is None:
                if expr.kws:
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
                    values = [self._cast_var(var, sigty)
                              for var, sigty in
                              zip(vars, signature.args[index])]
                    return cgutils.make_anonymous_struct(self.builder, values)

                argvals = typing.fold_arguments(pysig,
                                                pos_args, dict(expr.kws),
                                                normal_handler,
                                                default_handler,
                                                stararg_handler)

        if isinstance(fnty, types.ExternalFunction):
            # Handle a named external function
            self.debug_print("# external function")
            fndesc = funcdesc.ExternalFunctionDescriptor(
                fnty.symbol, fnty.sig.return_type, fnty.sig.args)
            func = self.context.declare_external_function(self.builder.module,
                                                          fndesc)
            res = self.context.call_external_function(
                self.builder, func, fndesc.argtypes, argvals)

        elif isinstance(fnty, types.NumbaFunction):
            # Handle a compiled Numba function
            self.debug_print("# calling numba function")
            res = self.context.call_internal(self.builder, fnty.fndesc,
                                             fnty.sig, argvals)

        elif isinstance(fnty, types.ExternalFunctionPointer):
            self.debug_print("# calling external function pointer")
            # Handle a C function pointer
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
                        obj = self.pyapi.from_native_value(gottyp, aval,
                                                           self.env_manager)
                        newargvals.append(obj)
                        pyvals.append(obj)
                    else:
                        newargvals.append(aval)

                # Call external function
                res = self.context.call_function_pointer(self.builder, pointer,
                                                         newargvals, fnty.cconv)
                # Release PyObjects
                for obj in pyvals:
                    self.pyapi.decref(obj)

                # Release the GIL
                self.pyapi.gil_release(gil_state)
            # If the external function pointer does NOT use libpython
            else:
                res = self.context.call_function_pointer(self.builder, pointer,
                                                         argvals, fnty.cconv)

        else:
            # Normal function resolution (for Numba-compiled functions)
            self.debug_print("# calling normal function: {0}".format(fnty))
            impl = self.context.get_function(fnty, signature)
            if signature.recvr:
                # The "self" object is passed as the function object
                # for bounded function
                the_self = self.loadvar(expr.func.name)
                # Prepend the self reference
                argvals = [the_self] + list(argvals)

            res = impl(self.builder, argvals)

            libs = getattr(impl, "libs", ())
            for lib in libs:
                self.library.add_linking_library(lib)

        return self.context.cast(self.builder, res, signature.return_type,
                                 resty)

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
            return self.lower_binop(resty, expr, expr.fn)
        elif expr.op == 'inplace_binop':
            lty = self.typeof(expr.lhs.name)
            if lty.mutable:
                return self.lower_binop(resty, expr, expr.fn)
            else:
                # inplace operators on non-mutable types reuse the same
                # definition as the corresponding copying operators.
                return self.lower_binop(resty, expr, expr.immutable_fn)
        elif expr.op == 'unary':
            val = self.loadvar(expr.value.name)
            typ = self.typeof(expr.value.name)
            # Get function
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.fn, signature)
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
            # If we have a tuple, we needn't do anything
            # (and we can't iterate over the heterogenous ones).
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
                    self.return_exception(ValueError)
                item = self.context.pair_first(self.builder,
                                               pair, pairty)
                tup = self.builder.insert_value(tup, item, i)

            # Call iternext() once more to check that the iterator
            # is exhausted.
            pair = iternext_impl(self.builder, (iterobj,))
            is_valid = self.context.pair_second(self.builder,
                                                pair, pairty)
            with cgutils.if_unlikely(self.builder, is_valid):
                self.return_exception(ValueError)

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
            signature = typing.signature(resty, self.typeof(expr.value.name),
                                         types.Const(expr.index))
            try:
                # Both get_function() and the returned implementation can
                # raise NotImplementedError if the types aren't supported
                impl = self.context.get_function("static_getitem", signature)
                return impl(self.builder, (self.loadvar(expr.value.name), expr.index))
            except NotImplementedError:
                if expr.index_var is None:
                    raise
                # Fall back on the generic getitem() implementation
                # for this type.
                signature = self.fndesc.calltypes[expr]
                return self.lower_getitem(resty, expr, expr.value,
                                          expr.index_var, signature)

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
            castvals = [self.context.cast(self.builder, val, fromty, resty.dtype)
                        for val, fromty in zip(itemvals, itemtys)]
            return self.context.build_list(self.builder, resty, castvals)

        elif expr.op == "build_set":
            # Insert in reverse order, as Python does
            items = expr.items[::-1]
            itemvals = [self.loadvar(i.name) for i in items]
            itemtys = [self.typeof(i.name) for i in items]
            castvals = [self.context.cast(self.builder, val, fromty, resty.dtype)
                        for val, fromty in zip(itemvals, itemtys)]
            return self.context.build_set(self.builder, resty, castvals)

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
            msg = ("Storing {value.type} to ptr of {ptr.type.pointee} ('{name}'). "
                   "FE type {fetype}").format(value=value, ptr=ptr,
                                              fetype=fetype, name=name)
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
        return cgutils.alloca_once(self.builder, lltype, name=name, zfill=True)

    def incref(self, typ, val):
        if not self.context.enable_nrt:
            return

        self.context.nrt_incref(self.builder, typ, val)

    def decref(self, typ, val):
        if not self.context.enable_nrt:
            return

        self.context.nrt_decref(self.builder, typ, val)
