from __future__ import print_function, division, absolute_import
import inspect
from collections import defaultdict
from llvm.core import Type, Builder, Module
import llvm.core as lc
from numba import ctypes_support as ctypes
from numba import ir, types, typing, cgutils, utils, config


try:
    import builtins
except ImportError:
    import __builtin__ as builtins


class LoweringError(Exception):
    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class FunctionDescriptor(object):
    def __init__(self, native, pymod, name, doc, blocks, typemap,
                 restype, calltypes, args, kws):
        self.native = native
        self.pymod = pymod
        self.name = name
        self.doc = doc
        self.blocks = blocks
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        self.argtypes = [self.typemap[a] for a in args]

        self.qualified_name = '.'.join([self.pymod.__name__, self.name])
        codedargs = '.'.join(str(a).replace(' ', '_') for a in self.argtypes)
        self.mangled_name = '.'.join([self.qualified_name, codedargs])


def _describe(interp):
    func = interp.bytecode.func
    fname = func.__name__
    pymod = inspect.getmodule(func)
    doc = func.__doc__ or ''
    args = interp.argspec.args
    kws = ()        # TODO
    return fname, pymod, doc, args, kws


def describe_function(interp, typemap, restype, calltypes):
    fname, pymod, doc, args, kws = _describe(interp)
    native = True
    sortedblocks = utils.SortedMap(utils.dict_iteritems(interp.blocks))
    fd = FunctionDescriptor(native, pymod, fname, doc, sortedblocks,
                            typemap, restype, calltypes, args, kws)
    return fd


def describe_pyfunction(interp):
    fname, pymod, doc, args, kws = _describe(interp)
    defdict = lambda: defaultdict(lambda: types.pyobject)
    typemap = defdict()
    restype = types.pyobject
    calltypes = defdict()
    native = False
    sortedblocks = utils.SortedMap(utils.dict_iteritems(interp.blocks))
    fd = FunctionDescriptor(native, pymod, fname, doc, sortedblocks,
                            typemap, restype,  calltypes, args, kws)
    return fd


class BaseLower(object):
    """
    Lower IR to LLVM
    """
    def __init__(self, context, fndesc):
        self.context = context
        self.fndesc = fndesc
        # Initialize LLVM
        self.module = Module.new("module.%s" % self.fndesc.name)

        # Install metadata
        md_pymod = cgutils.MetadataKeyStore(self.module, "python.module")
        md_pymod.set(fndesc.pymod.__name__)

        # Setup function
        self.function = context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)

        # Internal states
        self.blkmap = {}
        self.varmap = {}

        # Subclass initialization
        self.init()

    def init(self):
        pass

    def post_lower(self):
        """Called after all blocks are lowered
        """
        pass

    def lower(self):
        # Init argument variables
        fnargs = self.context.get_arguments(self.function)
        for ak, av in zip(self.fndesc.args, fnargs):
            at = self.typeof(ak)
            av = self.context.get_argument_value(self.builder, at, av)
            av = self.init_argument(av)
            self.storevar(av, ak)
        # Init blocks
        for offset in self.fndesc.blocks:
            bname = "B%d" % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)
        # Lower all blocks
        for offset, block in self.fndesc.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()
        # Close entry block
        self.builder.position_at_end(self.entry_block)
        self.builder.branch(self.blkmap[0])

        if config.DEBUG:
            print(self.module)
        self.module.verify()

    def init_argument(self, arg):
        return arg

    def lower_block(self, block):
        for inst in block.body:
            try:
                self.lower_inst(inst)
            except LoweringError:
                raise
            except Exception as e:
                msg = "Internal error:\n%s: %s" % (type(e).__name__, e)
                raise LoweringError(msg, inst.loc)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]


class Lower(BaseLower):
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
            assert cond.type == Type.int(1), ("cond is not i1: %s" % inst)
            self.builder.cbranch(cond, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Return):
            val = self.loadvar(inst.value.name)
            oty = self.typeof(inst.value.name)
            ty = self.fndesc.restype
            if isinstance(ty, types.Optional):
                if oty == types.none:
                    self.context.return_native_none(self.builder)
                    return
                else:
                    ty = ty.type

            if ty != oty:
                val = self.context.cast(self.builder, val, oty, ty)
            retval = self.context.get_return_value(self.builder, ty, val)
            self.context.return_value(self.builder, retval)

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
            assert targetty == signature.args[0]
            index = self.context.cast(self.builder, index, indexty,
                                      signature.args[1])
            value = self.context.cast(self.builder, value, valuety,
                                      signature.args[2])

            return impl(self.builder, (target, index, value))

        elif isinstance(inst, ir.Del):
            pass

        else:
            raise NotImplementedError(type(inst))

    def lower_assign(self, ty, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            if self.context.is_struct_type(ty):
                const = self.context.get_constant_struct(self.builder, ty,
                                                         value.value)
            else:
                const = self.context.get_constant(ty, value.value)
            return const

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            return self.context.cast(self.builder, val, oty, ty)

        elif isinstance(value, ir.Global):
            if (isinstance(ty, types.Dummy) or
                    isinstance(ty, types.Module) or
                    isinstance(ty, types.Function) or
                    isinstance(ty, types.Dispatcher)):
                return self.context.get_dummy_value()

            elif ty == types.boolean:
                return self.context.get_constant(ty, value.value)

            elif ty in types.number_domain:
                return self.context.get_constant(ty, value.value)

            elif isinstance(ty, types.Array):
                return self.context.make_constant_array(self.builder, ty,
                                                        value.value)

            else:
                raise NotImplementedError('global', ty)

        else:
            raise NotImplementedError(type(value), value)

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
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
            assert not expr.kws
            argvals = [self.loadvar(a.name) for a in expr.args]
            argtyps = [self.typeof(a.name) for a in expr.args]
            signature = self.fndesc.calltypes[expr]
            fnty = self.typeof(expr.func.name)
            castvals = [self.context.cast(self.builder, av, at, ft)
                        for av, at, ft in zip(argvals, argtyps,
                                              signature.args)]

            if isinstance(fnty, types.Method):
                # Method of objects are handled differently
                fnobj = self.loadvar(expr.func.name)
                res = self.context.call_class_method(self.builder, fnobj,
                                                     signature, castvals)

            elif isinstance(fnty, types.FunctionPointer):
                # Handle function pointer)
                pointer = fnty.funcptr
                res = self.context.call_function_pointer(self.builder, pointer,
                                                         signature, castvals)

            else:
                # Normal function resolution
                impl = self.context.get_function(fnty, signature)
                res = impl(self.builder, castvals)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op in ('getiter', 'iternext', 'itervalid', 'iternextsafe'):
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.op, signature)
            [fty] = signature.args
            castval = self.context.cast(self.builder, val, ty, fty)
            res = impl(self.builder, (castval,))
            return self.context.cast(self.builder, res, signature.return_type,
                                    resty)

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            impl = self.context.get_attribute(val, ty, expr.attr)
            if impl is None:
                # ignore the attribute
                res = self.context.get_dummy_value()
            else:
                res = impl(self.context, self.builder, ty, val)
                if not isinstance(impl.return_type, types.Kind):
                    res = self.context.cast(self.builder, res, impl.return_type,
                                            resty)
            return res

        elif expr.op == "getitem":
            baseval = self.loadvar(expr.target.name)
            indexval = self.loadvar(expr.index.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function("getitem", signature)
            argvals = (baseval, indexval)
            argtyps = (self.typeof(expr.target.name),
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
                tup = self.builder.insert_value(tup, itemvals[i], i)
            return tup

        raise NotImplementedError(expr)

    def getvar(self, name):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, self.typeof(name))
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        ptr = self.getvar(name)
        assert value.type == ptr.type.pointee, (str(value.type),
                                                str(ptr.type.pointee))
        self.builder.store(value, ptr)

    def alloca(self, name, type):
        ltype = self.context.get_value_type(type)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.position_at_end(bb)
        return ptr


PYTHON_OPMAP = {
     '+': "number_add",
     '-': "number_subtract",
     '*': "number_multiply",
    '/?': "number_divide",
     '/': "number_truedivide",
    '//': "number_floordivide",
     '%': "number_remainder",
    '**': "number_power",
    '<<': "number_lshift",
    '>>': "number_rshift",
     '&': "number_and",
     '|': "number_or",
     '^': "number_xor",
}


class PyLower(BaseLower):
    def init(self):
        self.pyapi = self.context.get_python_api(self.builder)

        # Add error handling block
        self.ehblock = self.function.append_basic_block('error')

    def post_lower(self):
        with cgutils.goto_block(self.builder, self.ehblock):
            self.cleanup()
            self.context.return_exc(self.builder)

    def init_argument(self, arg):
        self.incref(arg)
        return arg

    def lower_inst(self, inst):
        if isinstance(inst, ir.Assign):
            value = self.lower_assign(inst)
            self.storevar(value, inst.target.name)

        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.object_setitem(target, index, value)
            negone = lc.Constant.int_signextend(ok.type, -1)
            pred = self.builder.icmp(lc.ICMP_EQ, ok, negone)
            with cgutils.if_unlikely(self.builder, pred):
                self.return_exception_raised()

        elif isinstance(inst, ir.Return):
            retval = self.loadvar(inst.value.name)
            self.incref(retval)
            self.cleanup()
            self.context.return_value(self.builder, retval)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            if cond.type == Type.int(1):
                istrue = cond
            else:
                istrue = self.pyapi.object_istrue(cond)
            zero = lc.Constant.null(istrue.type)
            pred = self.builder.icmp(lc.ICMP_NE, istrue, zero)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]
            self.builder.cbranch(pred, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Del):
            obj = self.loadvar(inst.value)
            self.decref(obj)

        else:
            raise NotImplementedError(type(inst), inst)

    def lower_assign(self, inst):
        """
        The returned object must have a new reference
        """
        value = inst.value
        if isinstance(value, ir.Const):
            return self.lower_const(value.value)
        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            self.incref(val)
            return val
        elif isinstance(value, ir.Expr):
            return self.lower_expr(value)
        elif isinstance(value, ir.Global):
            return self.lower_global(value.name, value.value)
        else:
            raise NotImplementedError(type(value), value)

    def lower_expr(self, expr):
        if expr.op == 'binop':
            lhs = self.loadvar(expr.lhs.name)
            rhs = self.loadvar(expr.rhs.name)
            if expr.fn in PYTHON_OPMAP:
                fname = PYTHON_OPMAP[expr.fn]
                fn = getattr(self.pyapi, fname)
                res = fn(lhs, rhs)
            else:
                # Assume to be rich comparision
                res = self.pyapi.object_richcompare(lhs, rhs, expr.fn)
            self.check_error(res)
            return res
        elif expr.op == 'unary':
            value = self.loadvar(expr.value.name)
            if expr.fn == '-':
                res = self.pyapi.number_negative(value)
            elif expr.fn == 'not':
                res = self.pyapi.object_not(value)
                negone = lc.Constant.int_signextend(Type.int(), -1)
                err = self.builder.icmp(lc.ICMP_EQ, res, negone)
                with cgutils.if_unlikely(self.builder, err):
                    self.return_exception_raised()

                longval = self.builder.zext(res, self.pyapi.long)
                res = self.pyapi.bool_from_long(longval)
            elif expr.fn == '~':
                res = self.pyapi.number_invert(value)
            else:
                raise NotImplementedError(expr)
            self.check_error(res)
            return res
        elif expr.op == 'call':
            argvals = [self.loadvar(a.name) for a in expr.args]
            fn = self.loadvar(expr.func.name)
            if not expr.kws:
                # No keyword
                ret = self.pyapi.call_function_objargs(fn, argvals)
            else:
                # Have Keywords
                keyvalues = [(k, self.loadvar(v.name)) for k, v in expr.kws]
                args = self.pyapi.tuple_pack(argvals)
                kws = self.pyapi.dict_pack(keyvalues)
                ret = self.pyapi.call(fn, args, kws)
                self.decref(kws)
                self.decref(args)
            self.check_error(ret)
            return ret
        elif expr.op == 'getattr':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getattr_string(obj, expr.attr)
            self.check_error(res)
            return res
        elif expr.op == 'build_tuple':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.tuple_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'build_list':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.list_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'getiter':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getiter(obj)
            self.check_error(res)
            self.storevar(res, '$iter$' + expr.value.name)
            return self.pack_iter(res)
        elif expr.op == 'iternext':
            iterstate = self.loadvar(expr.value.name)
            iterobj, valid = self.unpack_iter(iterstate)
            item = self.pyapi.iter_next(iterobj)
            self.set_iter_valid(iterstate, item)
            return item
        elif expr.op == 'iternextsafe':
            iterstate = self.loadvar(expr.value.name)
            iterobj, _ = self.unpack_iter(iterstate)
            item = self.pyapi.iter_next(iterobj)
            # TODO need to add exception
            self.check_error(item)
            self.set_iter_valid(iterstate, item)
            return item
        elif expr.op == 'itervalid':
            iterstate = self.loadvar(expr.value.name)
            _, valid = self.unpack_iter(iterstate)
            return valid
        elif expr.op == 'getitem':
            target = self.loadvar(expr.target.name)
            index = self.loadvar(expr.index.name)
            res = self.pyapi.object_getitem(target, index)
            self.check_error(res)
            return res
        elif expr.op == 'getslice':
            target = self.loadvar(expr.target.name)
            start = self.loadvar(expr.start.name)
            stop = self.loadvar(expr.stop.name)

            slicefn = self.get_builtin_obj("slice")
            sliceobj = self.pyapi.call_function_objargs(slicefn, (start, stop))
            self.decref(slicefn)
            self.check_error(sliceobj)

            res = self.pyapi.object_getitem(target, sliceobj)
            self.check_error(res)

            return res
        else:
            raise NotImplementedError(expr)

    def lower_const(self, const):
        if isinstance(const, str):
            ret = self.pyapi.string_from_string_and_size(const)
            self.check_error(ret)
            return ret
        elif isinstance(const, complex):
            real = self.context.get_constant(types.float64, const.real)
            imag = self.context.get_constant(types.float64, const.imag)
            ret = self.pyapi.complex_from_doubles(real, imag)
            self.check_error(ret)
            return ret
        elif isinstance(const, float):
            fval = self.context.get_constant(types.float64, const)
            ret = self.pyapi.float_from_double(fval)
            self.check_error(ret)
            return ret
        elif isinstance(const, int):
            if utils.bit_length(const) >= 64:
                raise ValueError("Integer is too big to be lowered")
            ival = self.context.get_constant(types.intp, const)
            return self.pyapi.long_from_ssize_t(ival)
        elif isinstance(const, tuple):
            items = [self.lower_const(i) for i in const]
            return self.pyapi.tuple_pack(items)
        elif const is Ellipsis:
            return self.get_builtin_obj("Ellipsis")
        elif const is None:
            return self.pyapi.make_none()
        else:
            raise NotImplementedError(type(const))

    def lower_global(self, name, value):
        """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
        moddict = self.pyapi.get_module_dict()
        obj = self.pyapi.dict_getitem_string(moddict, name)
        self.incref(obj)  # obj is borrowed

        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block

            with cgutils.ifthen(self.builder, obj_is_null):
                mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
                builtin = self.builtin_lookup(mod, name)
                bbif = self.builder.basic_block

            retval = self.builder.phi(self.pyapi.pyobj)
            retval.add_incoming(obj, bbelse)
            retval.add_incoming(builtin, bbif)

        else:
            retval = obj
            with cgutils.if_unlikely(self.builder, self.is_null(retval)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

        self.incref(retval)
        return retval

    # -------------------------------------------------------------------------

    def get_builtin_obj(self, name):
        moddict = self.pyapi.get_module_dict()
        mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
        return self.builtin_lookup(mod, name)

    def builtin_lookup(self, mod, name):
        """
        Args
        ----
        mod:
            The __builtins__ dictionary or module
        name: str
            The object to lookup
        """
        fromdict = self.pyapi.dict_getitem_string(mod, name)
        self.incref(fromdict)       # fromdict is borrowed
        bbifdict = self.builder.basic_block

        with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
            # This happen if we are using the __main__ module
            frommod = self.pyapi.object_getattr_string(mod, name)

            with cgutils.if_unlikely(self.builder, self.is_null(frommod)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

            bbifmod = self.builder.basic_block

        builtin = self.builder.phi(self.pyapi.pyobj)
        builtin.add_incoming(fromdict, bbifdict)
        builtin.add_incoming(frommod, bbifmod)

        return builtin

    def pack_iter(self, obj):
        iterstate = PyIterState(self.context, self.builder)
        iterstate.iterator = obj
        iterstate.valid = cgutils.true_bit
        return iterstate._getvalue()

    def unpack_iter(self, state):
        iterstate = PyIterState(self.context, self.builder, value=state)
        return tuple(iterstate)

    def set_iter_valid(self, state, item):
        iterstate = PyIterState(self.context, self.builder, value=state)
        iterstate.valid = cgutils.is_not_null(self.builder, item)

        with cgutils.if_unlikely(self.builder, self.is_null(item)):
            self.check_occurred()

    def check_occurred(self):
        err_occurred = cgutils.is_not_null(self.builder,
                                           self.pyapi.err_occurred())

        with cgutils.if_unlikely(self.builder, err_occurred):
            self.return_exception_raised()

    def check_error(self, obj):
        with cgutils.if_unlikely(self.builder, self.is_null(obj)):
            self.return_exception_raised()

        return obj

    def is_null(self, obj):
        return cgutils.is_null(self.builder, obj)

    def return_exception_raised(self):
        self.builder.branch(self.ehblock)

    def return_error_occurred(self):
        self.cleanup()
        self.context.return_exc(self.builder)

    def getvar(self, name, ltype=None):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, ltype=ltype)
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        """
        Stores a llvm value and allocate stack slot if necessary.
        The llvm value can be of arbitrary type.
        """
        ptr = self.getvar(name, ltype=value.type)
        old = self.builder.load(ptr)
        assert value.type == ptr.type.pointee, (str(value.type),
                                                str(ptr.type.pointee))
        self.builder.store(value, ptr)
        # Safe to call decref even on non python object
        self.decref(old)

    def cleanup(self):
        for var in utils.dict_itervalues(self.varmap):
            self.decref(self.builder.load(var))

    def alloca(self, name, ltype=None):
        """
        Allocate a stack slot and initialize it to NULL.
        The default is to allocate a pyobject pointer.
        Use ``ltype`` to override.
        """
        if ltype is None:
            ltype = self.context.get_value_type(types.pyobject)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.store(cgutils.get_null_value(ltype), ptr)
        self.builder.position_at_end(bb)
        return ptr

    def incref(self, value):
        self.pyapi.incref(value)

    def decref(self, value):
        """
        This is allow to be called on non pyobject pointer, in which case
        no code is inserted.

        If the value is a PyIterState, it unpack the structure and decref
        the iterator.
        """
        lpyobj = self.context.get_value_type(types.pyobject)

        if value.type.kind == lc.TYPE_POINTER:
            if value.type != lpyobj:
                pass
                #raise AssertionError(value.type)
                # # Handle PyIterState
                # not_null = cgutils.is_not_null(self.builder, value)
                # with cgutils.if_likely(self.builder, not_null):
                #     iterstate = PyIterState(self.context, self.builder,
                #                             value=value)
                #     value = iterstate.iterator
                #     self.pyapi.decref(value)
            else:
                self.pyapi.decref(value)


class PyIterState(cgutils.Structure):
    _fields = [
        ("iterator", types.pyobject),
        ("valid",    types.boolean),
    ]
