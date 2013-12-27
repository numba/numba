from __future__ import print_function
from llvm.core import Type, Builder, Module
from numba import ir, utils


class FunctionDescriptor(object):
    def __init__(self, name, blocks, typemap, restype, args, kws):
        self.name = name
        self.blocks = blocks
        self.typemap = typemap
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        self.argtypes = [self.typemap[a] for a in args]


def describe_function(interp, typemap, restype):
    args = interp.argspec.args
    kws = ()            #TODO
    fd = FunctionDescriptor(interp.bytecode.func.__name__,
                            interp.blocks, typemap, restype, args, kws)
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

        argtypes = [self.context.get_argument_type(aty)
                    for aty in self.fndesc.argtypes]
        restype = self.context.get_return_type(self.fndesc.restype)
        fnty = Type.function(restype, argtypes)

        self.function = self.module.add_function(fnty, name=self.fndesc.name)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)

        # Internal states
        self.blkmap = {}
        self.varmap = {}

    def lower(self):
        # Init argument variables
        for ak, av in zip(self.fndesc.args, self.function.args):
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
                val = self.context.cast(val, oty, ty)
            retval = self.context.get_return_value(self.builder, ty, val)
            self.builder.ret(retval)
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
            impl = self.context.get_function(expr.fn, lty, rty)
            assert impl.signature.return_type == resty
            lhs = self.loadvar(lhs.name)
            rhs = self.loadvar(rhs.name)
            # Convert argument to match
            return impl(self.context, self.builder, (lhs, rhs))
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

