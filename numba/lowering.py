from __future__ import print_function
import inspect
from llvm.core import Type, Builder, Module
from numba import ir, types, typing


class FunctionDescriptor(object):
    def __init__(self, pymod, name, doc, blocks, typemap, restype, calltypes,
                 args,
                 kws):
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


def describe_function(interp, typemap, restype, calltypes):
    func = interp.bytecode.func
    pymod = inspect.getmodule(func)
    doc = func.__doc__ or ''
    args = interp.argspec.args
    kws = ()            #TODO
    fd = FunctionDescriptor(pymod, interp.bytecode.func.__name__, doc,
                            interp.blocks, typemap, restype, calltypes, args,
                            kws)
    return fd


class Lower(object):
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

    def lower(self):
        # Init argument variables
        fnargs = self.context.get_arguments(self.function)
        for ak, av in zip(self.fndesc.args, fnargs):
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
        # Close entry block
        self.builder.position_at_end(self.entry_block)
        self.builder.branch(self.blkmap[0])

        print(self.module)
        self.module.verify()


    def lower_block(self, block):
        for inst in block.body:
            self.lower_inst(inst)

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
            if isinstance(ty, types.Dummy):
                return self.context.get_dummy_value()
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

    def typeof(self, varname):
        return self.fndesc.typemap[varname]

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

