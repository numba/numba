import inspect
import operator
from llvm import core as lc

from .errors import error_context
from . import types, typesets

class CodeGen(object):
    def __init__(self, func, blocks, args, return_type, implib):
        self.func = func
        self.argspec = inspect.getargspec(func)
        assert not self.argspec.keywords
        assert not self.argspec.defaults
        assert not self.argspec.varargs

        self.blocks = blocks
        self.args = args
        self.return_type = return_type
        self.implib = implib

    def make_module(self):
        return lc.Module.new('module.%s' % self.func.__name__)

    def make_function(self):
        largtys = []
        for argname in self.argspec.args:
            argtype = self.args[argname]
            argty = argtype.llvm_as_argument()
            largtys.append(argty)

        if self.return_type != types.void:
            lretty = self.return_type.llvm_as_return()
            fnty_args = largtys + [lretty]
        else:
            fnty_args = largtys

        lfnty = lc.Type.function(lc.Type.void(), fnty_args)
        lfunc = self.lmod.add_function(lfnty, name=self.func.__name__)

        return lfunc

    def codegen(self):
        self.lmod = self.make_module()
        self.lfunc = self.make_function()

        # initialize all blocks
        self.bbmap = {}
        for block in self.blocks:
            bb = self.lfunc.append_basic_block('B%d' % block.offset)
            self.bbmap[block] = bb


        self.builder = lc.Builder.new(self.bbmap[self.blocks[0]])
        # initialize stack storage
        varnames = {}
        for block in self.blocks:
            for inst in block.code:
                if inst.opcode == 'store':
                    if inst.name in varnames:
                        if inst.astype != varnames[inst.name]:
                            raise AssertionError('store type mismatch')
                    else:
                        varnames[inst.name] = inst.astype
        self.alloca = {}
        for name, vtype in varnames.iteritems():
            storage = self.builder.alloca(vtype.llvm_as_value(),
                                          name='var.' + name)
            self.alloca[name] = storage

        # generate all instructions
        self.valmap = {}
        for block in self.blocks:
            self.builder.position_at_end(self.bbmap[block])
            for inst in block.code:
                with error_context(lineno=inst.lineno,
                                   when='instruction codegen'):
                    self.valmap[inst] = self.op(inst)
            self.op(block.terminator)


    def cast(self, val, src, dst):
        if src == dst:
            return val
        else:
            return src.llvm_cast(self.builder, val, dst)

    def op(self, inst):
        attr = 'op_%s' % inst.opcode
        func = getattr(self, attr, self.generic_op)
        return func(inst)

    def generic_op(self, inst):
        print self.lfunc
        raise NotImplementedError(inst)

    def op_branch(self, inst):
        cond = self.cast(self.valmap[inst.cond], inst.cond.type, types.boolean)
        null = lc.Constant.null(cond.type)
        pred = self.builder.icmp(lc.ICMP_NE, cond, null)
        bbtrue = self.bbmap[inst.truebr]
        bbfalse = self.bbmap[inst.falsebr]
        self.builder.cbranch(pred, bbtrue, bbfalse)

    def op_jump(self, inst):
        self.builder.branch(self.bbmap[inst.target])

    def op_ret(self, inst):
        val = self.cast(self.valmap[inst.value], inst.value.type, inst.astype)
        self.builder.store(val, self.lfunc.args[-1])
        self.builder.ret_void()

    def op_arg(self, inst):
        argval = self.lfunc.args[inst.num]
        return inst.type.llvm_value_from_arg(self.builder, argval)

    def op_store(self, inst):
        val = self.valmap[inst.value]
        val = self.cast(val, inst.value.type, inst.astype)
        self.builder.store(val, self.alloca[inst.name])

    def op_load(self, inst):
        storage = self.alloca[inst.name]
        return self.builder.load(storage)

    def op_call(self, inst):
        imp = self.implib.get(inst.defn)
        assert not inst.kws
        args = [self.cast(self.valmap[aval], aval.type, atype)
                for aval, atype in zip(inst.args, imp.args)]
        return imp(self.builder, args)

    def op_const(self, inst):
        return inst.type.llvm_const(self.builder, inst.value)

    def op_global(self, inst):
        if inst.type == types.function_type:
            return # do nothing
        else:
            assert False

    def op_phi(self, inst):
        values = inst.phi.values()
        if len(values) == 1:
            return self.valmap[values[0]]
        assert False

#-----------------------------------------------------------------------------
# Function Implementation

class ImpLib(object):
    def __init__(self, funclib):
        self.funclib = funclib
        self.implib = {}

    def define(self, imp):
        defn = self.funclib.lookup(imp.funcobj, imp.args)
        if defn.return_type != imp.return_type:
            msg = ('return-type mismatch for implementation; '
                   'expect %s but got %s')
            raise TypeError(msg % (defn.return_type, imp.return_type))
        self.implib[defn] = imp

    def get(self, funcdef):
        return self.implib[funcdef]

    def populate_builtin(self):
        populate_builtin_impl(self)

class Imp(object):
    def __init__(self, impl, funcobj, args, return_type):
        self.impl = impl
        self.funcobj = funcobj
        self.args = args
        self.return_type = return_type

    def __call__(self, builder, args):
        return self.impl(builder, args)

def imp_eq_signed(builder, args):
    a, b = args
    return builder.icmp(lc.ICMP_EQ, a, b)

def imp_add_integer(builder, args):
    a, b = args
    return builder.add(a, b)

def imp_range(builder, args):
    assert len(args) == 1
    (stop,) = args

    start = types.intp.llvm_const(builder, 0)
    step = types.intp.llvm_const(builder, 1)

    rangetype = types.range_type.llvm_as_value()
    rangeobj = lc.Constant.undef(rangetype)

    rangeobj = builder.insert_value(rangeobj, start, 0)
    rangeobj = builder.insert_value(rangeobj, stop, 1)
    rangeobj = builder.insert_value(rangeobj, step, 2)

    return rangeobj

def imp_range_iter(builder, args):
    obj, = args
    entry = builder.basic_block.function.basic_blocks[0]
    cur = builder.basic_block
    # allocate at the beginning
    # assuming a range object must be used statically
    builder.position_at_beginning(entry)
    ptr = builder.alloca(obj.type)
    builder.position_at_end(cur)
    builder.store(obj, ptr)
    return ptr

def imp_range_valid(builder, args):
    ptr, = args
    idx0 = types.int32.llvm_const(builder, 0)
    idx1 = types.int32.llvm_const(builder, 1)
    start = builder.load(builder.gep(ptr, [idx0, idx0]))
    stop = builder.load(builder.gep(ptr, [idx0, idx1]))
    return builder.icmp(lc.ICMP_ULT, start, stop)

def imp_range_next(builder, args):
    ptr, = args
    idx0 = types.int32.llvm_const(builder, 0)
    idx2 = types.int32.llvm_const(builder, 2)
    startptr = builder.gep(ptr, [idx0, idx0])
    start = builder.load(startptr)
    step = builder.load(builder.gep(ptr, [idx0, idx2]))
    next = builder.add(start, step)
    builder.store(next, startptr)
    return start

def bool_op_imp(funcobj, imp, typeset):
    return [Imp(imp, funcobj, args=(ty, ty), return_type=types.boolean)
            for ty in typeset]

def binary_op_imp(funcobj, imp, typeset):
    return [Imp(imp, funcobj, args=(ty, ty), return_type=ty)
            for ty in typeset]

def populate_builtin_impl(implib):
    imps = []

    imps += binary_op_imp(operator.add, imp_add_integer, typesets.integer_set)

    imps += bool_op_imp(operator.eq, imp_eq_signed, typesets.signed_set)

    imps += [Imp(imp_range, range,
                 args=(types.intp,),
                 return_type=types.range_type)]

    imps += [Imp(imp_range_iter, iter,
                 args=(types.range_type,),
                 return_type=types.range_iter_type)]

    imps += [Imp(imp_range_valid, 'itervalid',
                 args=(types.range_iter_type,),
                 return_type=types.boolean)]

    imps += [Imp(imp_range_next, 'iternext',
                 args=(types.range_iter_type,),
                 return_type=types.intp)]

    for imp in imps:
        implib.define(imp)

