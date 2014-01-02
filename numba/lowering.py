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
        self.native = False
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
    fd = FunctionDescriptor(native, pymod, fname, doc, interp.blocks,
                            typemap, restype, calltypes, args, kws)
    return fd


def describe_pyfunction(interp):
    fname, pymod, doc, args, kws = _describe(interp)
    defdict = lambda: defaultdict(lambda: types.pyobject)
    typemap = defdict()
    restype = types.pyobject
    calltypes = defdict()
    native = False
    fd = FunctionDescriptor(native, pymod, fname, doc, interp.blocks,
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
            impl = self.context.get_function(fnty, signature)
            castvals = [self.context.cast(self.builder, av, at, ft)
                        for av, at, ft in zip(argvals, argtyps,
                                              signature.args)]
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
        assert value.type == ptr.type.pointee
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
        scope = self.context.get_scope(self.function)
        self.pyapi = self.context.get_python_api(self.builder, scope)

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
            istrue = self.pyapi.object_istrue(cond)
            zero = lc.Constant.null(istrue.type)
            pred = self.builder.icmp(lc.ICMP_NE, istrue, zero)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]
            self.builder.cbranch(pred, tr, fl)
        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)
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
        else:
            raise NotImplementedError(type(const))

    def lower_global(self, name):
        obj = self.pyapi.dict_getitem_string(self.pyapi.globalscope, name)

        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block

            with cgutils.ifthen(self.builder, obj_is_null):
                mod = self.pyapi.dict_getitem_string(self.pyapi.globalscope,
                                                     "__builtins__")
                fromdict = self.pyapi.dict_getitem_string(mod, name)

                bbifdict = self.builder.basic_block

                with cgutils.ifthen(self.builder, self.is_null(fromdict)):
                    # This happen if we are using the __main__ module
                    frommod = self.pyapi.object_getattr_string(mod, name)
                    self.check_error(frommod)
                    bbifmod = self.builder.basic_block

                builtin = self.builder.phi(self.pyapi.pyobj)
                builtin.add_incoming(fromdict, bbifdict)
                builtin.add_incoming(frommod, bbifmod)
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

    def check_error(self, obj):
        bbold = self.builder.basic_block
        with cgutils.ifthen(self.builder, self.is_null(obj)):
            self.builder.branch(self.ehblock)

        # Assign branch weight
        lastinstr = bbold.instructions[-1]

        mdid = lc.MetaDataString.get(self.module, "branch_weights")
        trueweight = lc.Constant.int(Type.int(), 99)
        falseweight = lc.Constant.int(Type.int(), 1)
        md = lc.MetaData.get(self.module, [mdid, trueweight, falseweight])

        lastinstr.set_metadata("prof", md)
        return obj

    def is_null(self, obj):
        null = self.context.get_constant_null(types.pyobject)
        return self.builder.icmp(lc.ICMP_EQ, obj, null)

    def return_error_occurred(self):
        self.cleanup()
        # TODO
        self.builder.unreachable()

    def getvar(self, name):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name)
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        ptr = self.getvar(name)
        old = self.builder.load(ptr)
        assert value.type == ptr.type.pointee
        self.builder.store(value, ptr)
        self.decref(old)

    def cleanup(self):
        for var in self.varmap.itervalues():
            self.decref(self.builder.load(var))

    def alloca(self, name):
        """
        Allocate a PyObject stack slot and initialize it to NULL
        """
        ltype = self.context.get_value_type(types.pyobject)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.store(self.context.get_constant_null(types.pyobject), ptr)
        self.builder.position_at_end(bb)
        return ptr

    def incref(self, value):
        self.pyapi.incref(value)

    def decref(self, value):
        self.pyapi.decref(value)
