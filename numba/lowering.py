from __future__ import print_function
import inspect
from collections import defaultdict
from llvm.core import Type, Builder, Module
import llvm.core as lc
from numba import ir, types, typing, cgutils, utils, DEBUG


try:
    import builtins
except ImportError:
    import __builtin__ as builtins


class FunctionDescriptor(object):
    def __init__(self, native, pymod, name, doc, blocks, typemap,
                 restype, calltypes, args, kws):
        # FIXME self.native is currently unused
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
    sortedblocks = utils.SortedMap(interp.blocks.iteritems())
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
    sortedblocks = utils.SortedMap(interp.blocks.iteritems())
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

        if DEBUG:
            print(self.module)
        self.module.verify()

    def init_argument(self, arg):
        return arg

    def lower_block(self, block):
        for inst in block.body:
            self.lower_inst(inst)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]


class Lower(BaseLower):
    def lower_inst(self, inst):
        if isinstance(inst, ir.Assign):
            ty = self.typeof(inst.target.name)
            val = self.lower_assign(ty, inst)
            self.storevar(val, inst.target.name)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]
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

            signature = typing.signature(types.none, targetty, indexty,
                                         valuety)

            impl = self.context.get_function("setitem", signature)

            argvals = (target, index, value)
            argtyps = (targetty, indexty, valuety)

            castvals = [self.context.cast(self.builder, av, at, ft)
                        for av, at, ft in zip(argvals, argtyps,
                                              signature.args)]

            return impl(self.context, self.builder, argtyps, castvals)

        elif isinstance(inst, ir.Del):
            pass

        else:
            raise NotImplementedError(type(inst))

    def lower_assign(self, ty, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            return self.context.get_constant(ty, value.value)

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Phi):
            return self.lower_phi(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            return self.context.cast(self.builder, val, oty, ty)

        elif isinstance(value, ir.Global):
            if (isinstance(ty, types.Dummy) or isinstance(ty, types.Module) or
                    isinstance(ty, types.Function)):
                return self.context.get_dummy_value()

            elif ty in types.number_domain:
                return self.context.get_constant(ty, value.value)

            else:
                raise NotImplementedError('global', ty)

        else:
            raise NotImplementedError(type(value), value)

    def lower_phi(self, ty, phi):
        valty = self.context.get_value_type(ty)
        ptrty = Type.pointer(valty)
        phinode = self.builder.phi(ptrty)
        for ib, iv in phi:
            ib = self.blkmap[ib]
            iv = self.getvar(iv.name)
            assert phinode.type == iv.type
            phinode.add_incoming(iv, ib)
        return self.builder.load(phinode)

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
            return impl(self.context, self.builder, signature.args, (lhs, rhs))
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
                return self.context.call_class_method(self.builder, fnobj,
                                                      signature.return_type,
                                                      argtyps, castvals)

            else:
                # Normal function resolution
                impl = self.context.get_function(fnty, signature)
                return impl(self.context, self.builder, argtyps, castvals)

        elif expr.op in ('getiter', 'iternext', 'itervalid'):
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.op, signature)
            (fty,) = signature.args
            castval = self.context.cast(self.builder, val, ty, fty)
            return impl(self.context, self.builder, (fty,), (castval,))

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            impl = self.context.get_attribute(val, ty, expr.attr)
            if impl is None:
                # ignore the attribute
                return self.context.get_dummy_value()
            else:
                return impl(self.context, self.builder, ty, val)

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
            return impl(self.context, self.builder, argtyps, castvals)

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
            return self.lower_global(value.name)
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
            else:
                raise NotImplementedError(expr)
            self.check_error(res)
            return res
        elif expr.op == 'call':
            assert not expr.kws
            argvals = [self.loadvar(a.name) for a in expr.args]
            fn = self.loadvar(expr.func.name)
            ret = self.pyapi.call_function_objargs(fn, argvals)
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
        elif expr.op == 'getiter':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getiter(obj)
            self.check_error(res)
            return self.pack_iter(res)
        elif expr.op == 'iternext':
            iterstate = self.loadvar(expr.value.name)
            iterobj, _ = self.unpack_iter(iterstate)
            item = self.pyapi.iter_next(iterobj)
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

    def lower_global(self, name):
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
            self.check_error(retval)

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
            self.check_error(frommod)
            bbifmod = self.builder.basic_block

        builtin = self.builder.phi(self.pyapi.pyobj)
        builtin.add_incoming(fromdict, bbifdict)
        builtin.add_incoming(frommod, bbifmod)

        return builtin

    def pack_iter(self, obj):
        iterstate = PyIterState(self.context, self.builder)
        iterstate.iterator = obj
        iterstate.valid = cgutils.false_bit
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
            self.builder.branch(self.ehblock)

    def check_error(self, obj):
        with cgutils.if_unlikely(self.builder, self.is_null(obj)):
            self.builder.branch(self.ehblock)

        return obj

    def is_null(self, obj):
        return cgutils.is_null(self.builder, obj)

    def return_error_occurred(self):
        self.cleanup()
        # TODO
        self.builder.unreachable()

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
        for var in self.varmap.itervalues():
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
                # Handle PyIterState
                not_null = cgutils.is_not_null(self.builder, value)
                with cgutils.if_likely(self.builder, not_null):
                    iterstate = PyIterState(self.context, self.builder,
                                            value=value)
                    value = iterstate.iterator
                    self.pyapi.decref(value)
            else:
                self.pyapi.decref(value)


class PyIterState(cgutils.Structure):
    _fields = [
        ("iterator", types.pyobject),
        ("valid",    types.boolean),
    ]
