from __future__ import print_function, division, absolute_import

import sys

from llvmlite.llvmpy.core import Constant, Type, Builder


from . import (_dynfunc, cgutils, config, funcdesc, generators, ir, types,
               typing, utils)


class LoweringError(Exception):
    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class ForbiddenConstruct(LoweringError):
    pass


class BaseLower(object):
    """
    Lower IR to LLVM
    """
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
        self.env = _dynfunc.Environment(
            globals=self.fndesc.lookup_module().__dict__)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.firstblk = min(self.blocks.keys())
        self.loc = -1

        # Subclass initialization
        self.init()

    def init(self):
        pass

    def pre_lower(self):
        """
        Called before lowering all blocks.
        """

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

        # Materialize LLVM Module
        self.library.add_ir_module(self.module)

    def extract_function_arguments(self):
        rawfnargs = self.call_conv.get_arguments(self.function)
        arginfo = self.context.get_arg_packer(self.fndesc.argtypes)
        self.fnargs = arginfo.from_arguments(self.builder, rawfnargs)
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

        # Run target specific post lowering transformation
        self.context.post_lowering(self.function)

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
                                                self.call_helper,
                                                release_gil=release_gil)
        self.context.create_cpython_wrapper(self.library, self.fndesc,
                                            self.call_helper,
                                            release_gil=release_gil)

    def setup_function(self, fndesc):
        # Setup function
        self.function = self.context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)
        self.call_helper = self.call_conv.init_call_helper(self.builder)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]


class Lower(BaseLower):

    GeneratorLower = generators.GeneratorLower

    def lower_inst(self, inst):
        if config.DEBUG_JIT:
            self.context.debug_print(self.builder, str(inst))
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

        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            index = self.loadvar(inst.index.name)

            targetty = self.typeof(inst.target.name)
            valuety = self.typeof(inst.value.name)
            indexty = self.typeof(inst.index.name)

            signature = self.fndesc.calltypes[inst]
            assert signature is not None
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

        elif isinstance(inst, ir.Del):
            pass

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

        elif isinstance(inst, ir.Raise):
            self.lower_raise(inst)

        else:
            raise NotImplementedError(type(inst))

    def lower_raise(self, inst):
        if inst.exception is None:
            # Reraise
            self.return_exception(None)
        else:
            exctype = self.typeof(inst.exception.name)
            if isinstance(exctype, types.ExceptionInstance):
                # raise <instance> => find the instantiation site
                excdef = self.interp.get_definition(inst.exception)
                if (not isinstance(excdef, ir.Expr) or excdef.op != 'call'
                    or excdef.kws):
                    raise NotImplementedError("unsupported kind of raising")
                # Try to infer the args tuple
                args = tuple(self.interp.get_definition(arg).infer_constant()
                             for arg in excdef.args)
            elif isinstance(exctype, types.ExceptionType):
                args = None
            else:
                raise NotImplementedError("cannot raise value of type %s"
                                          % (exctype,))
            self.return_exception(exctype.exc_class, args)

    def lower_assign(self, ty, inst):
        value = inst.value
        # In nopython mode, closure vars are frozen like globals
        if isinstance(value, (ir.Const, ir.Global, ir.FreeVar)):
            if isinstance(ty, types.ExternalFunctionPointer):
                return self.context.get_constant_generic(self.builder, ty,
                                                         value.value)

            elif (isinstance(ty, types.Dummy) or
                    isinstance(ty, types.Module) or
                    isinstance(ty, types.Function) or
                    isinstance(ty, types.Dispatcher)):
                return self.context.get_dummy_value()

            elif isinstance(ty, types.Array):
                return self.context.make_constant_array(self.builder, ty,
                                                        value.value)

            else:
                return self.context.get_constant_generic(self.builder, ty,
                                                         value.value)

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            return self.context.cast(self.builder, val, oty, ty)

        elif isinstance(value, ir.Arg):
            return self.fnargs[value.index]

        elif isinstance(value, ir.Yield):
            return self.lower_yield(ty, value)

        else:
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

    def lower_binop(self, resty, expr):
        lhs = expr.lhs
        rhs = expr.rhs
        lty = self.typeof(lhs.name)
        rty = self.typeof(rhs.name)
        lhs = self.loadvar(lhs.name)
        rhs = self.loadvar(rhs.name)
        # Get function
        signature = self.fndesc.calltypes[expr]
        impl = self.context.get_function(expr.fn, signature)
        # Convert argument to match
        lhs = self.context.cast(self.builder, lhs, lty, signature.args[0])
        rhs = self.context.cast(self.builder, rhs, rty, signature.args[1])
        res = impl(self.builder, (lhs, rhs))
        return self.context.cast(self.builder, res, signature.return_type,
                                 resty)

    def lower_call(self, resty, expr):
        signature = self.fndesc.calltypes[expr]
        if isinstance(signature.return_type, types.Phantom):
            return self.context.get_dummy_value()

        if isinstance(expr.func, ir.Intrinsic):
            fnty = expr.func.name
            argvals = expr.func.args
        else:
            fnty = self.typeof(expr.func.name)

            # Fold keyword arguments and resolve default argument values
            defargs = set()
            try:
                pysig = fnty.pysig
            except AttributeError:
                if expr.kws:
                    raise NotImplementedError("unsupported keyword arguments "
                                              "when calling %s" % (fnty,))
                args = expr.args
            else:
                ba = pysig.bind(*expr.args, **dict(expr.kws))
                for i, param in enumerate(pysig.parameters.values()):
                    name = param.name
                    default = param.default
                    if (default is not param.empty and
                        name not in ba.arguments):
                        value = self.context.get_constant_generic(
                            self.builder, signature.args[i], default)
                        ba.arguments[name] = value
                        defargs.add(i)
                # This is ensured in Dispatcher.get_call_template()
                assert not ba.kwargs
                args = ba.args

            # Fetch and cast non-default arguments
            argvals = list(args)
            for i, a in enumerate(argvals):
                if i not in defargs:
                    ty = self.typeof(a.name)
                    val = self.loadvar(a.name)
                    a = self.context.cast(self.builder, val, ty,
                                          signature.args[i])
                    argvals[i] = a

        if isinstance(fnty, types.ExternalFunction):
            # Handle a named external function
            fndesc = funcdesc.ExternalFunctionDescriptor(
                fnty.symbol, fnty.sig.return_type, fnty.sig.args)
            func = self.context.declare_external_function(
                    cgutils.get_module(self.builder), fndesc)
            res = self.context.call_external_function(
                self.builder, func, fndesc.argtypes, argvals)

        elif isinstance(fnty, types.Method):
            # Method of objects are handled differently
            fnobj = self.loadvar(expr.func.name)
            res = self.context.call_class_method(self.builder, fnobj,
                                                 signature, argvals)

        elif isinstance(fnty, types.ExternalFunctionPointer):
            # Handle a C function pointer
            pointer = self.loadvar(expr.func.name)
            res = self.context.call_function_pointer(self.builder, pointer,
                                                     argvals, fnty.cconv)

        else:
            # Normal function resolution (for Numba-compiled functions)
            impl = self.context.get_function(fnty, signature)
            if signature.recvr:
                # The "self" object is passed as the function object
                # for bounded function
                the_self = self.loadvar(expr.func.name)
                # Prepend the self reference
                argvals = [the_self] + argvals

            res = impl(self.builder, argvals)
            libs = getattr(impl, "libs", ())
            for lib in libs:
                self.library.add_linking_library(lib)
        return self.context.cast(self.builder, res, signature.return_type,
                                 resty)

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
            return self.lower_binop(resty, expr)
        elif expr.op == 'inplace_binop':
            lty = self.typeof(expr.lhs.name)
            if not lty.mutable:
                # inplace operators on non-mutable types reuse the same
                # definition as the corresponding copying operators.
                return self.lower_binop(resty, expr)
        elif expr.op == 'unary':
            val = self.loadvar(expr.value.name)
            typ = self.typeof(expr.value.name)
            # Get function
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.fn, signature)
            # Convert argument to match
            val = self.context.cast(self.builder, val, typ, signature.args[0])
            res = impl(self.builder, [val])
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'call':
            return self.lower_call(resty, expr)

        elif expr.op == 'pair_first':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            return self.context.pair_first(self.builder, val, ty)

        elif expr.op == 'pair_second':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            return self.context.pair_second(self.builder, val, ty)

        elif expr.op in ('getiter', 'iternext'):
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.op, signature)
            [fty] = signature.args
            castval = self.context.cast(self.builder, val, ty, fty)
            res = impl(self.builder, (castval,))
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'exhaust_iter':
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            # If we have a heterogenous tuple, we needn't do anything,
            # and we can't iterate over it anyway.
            if isinstance(ty, types.Tuple):
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

            return tup

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)

            if isinstance(resty, types.BoundFunction):
                # if we are getting out a method, assume we have typed this
                # properly and just build a bound function object
                res = self.context.get_bound_function(self.builder, val, ty)
            else:
                impl = self.context.get_attribute(val, ty, expr.attr)

                if impl is None:
                    # ignore the attribute
                    res = self.context.get_dummy_value()
                else:
                    res = impl(self.context, self.builder, ty, val, expr.attr)
            return res

        elif expr.op == "static_getitem":
            baseval = self.loadvar(expr.value.name)
            indexval = self.context.get_constant(types.intp, expr.index)
            if cgutils.is_struct(baseval.type):
                # Statically extract the given element from the structure
                # (structures aren't dynamically indexable).
                return self.builder.extract_value(baseval, expr.index)
            else:
                # Fall back on the generic getitem() implementation
                # for this type.
                signature = typing.signature(resty,
                                             self.typeof(expr.value.name),
                                             types.intp)
                impl = self.context.get_function("getitem", signature)
                argvals = (baseval, indexval)
                res = impl(self.builder, argvals)
                return self.context.cast(self.builder, res, signature.return_type,
                                         resty)

        elif expr.op == "getitem":
            baseval = self.loadvar(expr.value.name)
            indexval = self.loadvar(expr.index.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function("getitem", signature)
            argvals = (baseval, indexval)
            argtyps = (self.typeof(expr.value.name),
                       self.typeof(expr.index.name))
            castvals = [self.context.cast(self.builder, av, at, ft)
                        for av, at, ft in zip(argvals, argtyps,
                                              signature.args)]
            res = impl(self.builder, castvals)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == "build_tuple":
            itemvals = [self.loadvar(i.name) for i in expr.items]
            itemtys = [self.typeof(i.name) for i in expr.items]
            castvals = [self.context.cast(self.builder, val, fromty, toty)
                        for val, toty, fromty in zip(itemvals, resty, itemtys)]
            tup = self.context.get_constant_undef(resty)
            for i in range(len(castvals)):
                tup = self.builder.insert_value(tup, castvals[i], i)
            return tup

        elif expr.op == "cast":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            castval = self.context.cast(self.builder, val, ty, resty)
            return castval

        raise NotImplementedError(expr)

    def getvar(self, name):
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        if name not in self.varmap:
            self.varmap[name] = self.alloca_lltype(name, value.type)
        ptr = self.getvar(name)
        assert value.type == ptr.type.pointee,\
            "store %s to ptr of %s" % (value.type, ptr.type.pointee)
        self.builder.store(value, ptr)

    def alloca(self, name, type):
        lltype = self.context.get_value_type(type)
        return self.alloca_lltype(name, lltype)

    def alloca_lltype(self, name, lltype):
        return cgutils.alloca_once(self.builder, lltype, name=name)
