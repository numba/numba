import inspect
from contextlib import contextmanager
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
