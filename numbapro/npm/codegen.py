import inspect, collections
from llvm import core as lc
from .errors import error_context
from . import types, macro, cgutils

codegen_context = collections.namedtuple('codegen_context',
                                         ['imp', 'builder', 'raises', 'lineno',
                                          'flags', 'cast'])
exception_info = collections.namedtuple('exception_info',
                                        ['exc', 'line'])

SUPPORTED_FLAGS = frozenset(['overflow',
                             'zerodivision',
                             'boundcheck',
                             'wraparound',])

def _check_supported_flags(flags):
    for f in flags:
        if f not in SUPPORTED_FLAGS:
            raise NameError('unsupported compiler flag: %s' % f)

def _expand_flags(flags):
    if 'no-exceptions' in flags:
        flags.discard('overflow')
        flags.discard('zerodivision')
        flags.discard('boundcheck')
        flags.discard('no-exceptions')

class Flags(object):
    def __init__(self, flags):
        flags = set(flags)
        _expand_flags(flags)
        _check_supported_flags(flags)
        self.flags = frozenset(flags)

    def __getattr__(self, k):
        if k not in SUPPORTED_FLAGS:
            raise AttributeError(k)
        else:
            return k in self.flags

class CodeGen(object):
    def __init__(self, name, func, blocks, args, return_type, implib, flags=()):
        self.name = name
        self.argspec = inspect.getargspec(func)
        assert not self.argspec.keywords
        assert not self.argspec.defaults
        assert not self.argspec.varargs

        self.blocks = blocks
        self.args = args
        self.return_type = return_type
        self.implib = implib
        self.exceptions = {}    # map errcode to exceptions
        self.flags = Flags(flags)
        self.external_modules = set()

    def make_module(self):
        return lc.Module.new('module.%s' % self.name)

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

        lfnty = lc.Type.function(lc.Type.int(), fnty_args)
        lfunc = self.lmod.add_function(lfnty, name=self.name)

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
                                   during='instruction codegen'):
                    self.valmap[inst] = self.op(inst)
            with error_context(lineno=inst.lineno,
                               during='instruction codegen'):
                self.op(block.terminator)

        # link external modules
        for m in self.external_modules:
            self.lmod.link_in(m, preserve=True)

    def add_exception(self, excobj, msg=None):
        errcode = len(self.exceptions) + 1
        lineno = self.imp_context.lineno
        if msg is not None:
            excobj = excobj('at line %d: %s' % (lineno, msg))
        self.exceptions[errcode] = exception_info(excobj, lineno)
        return errcode

    def raises(self, excobj, msg=None):
        self.return_error(self.add_exception(excobj, msg=msg))

    def cast(self, val, src, dst):
        if src == dst:                          # types are the same
            return val
        elif isinstance(dst, types.Kind):       # cast to generic type
            if not dst.matches(src):
                raise TypeError('kind mismatch: expect %s got %s' %
                                (dst, src))
            return val
        elif not self.flags.overflow:            # use unguarded cast
            return src.llvm_cast(self.builder, val, dst)
        else:                                   # use guarded cast
            return src.llvm_cast_guarded(self.builder, self.raises, val, dst)

    def return_error(self, errcode):
        '''errcode is a python integer
        '''
        assert errcode != 0
        retty = self.lfunc.type.pointee.return_type
        self.builder.ret(lc.Constant.int(retty, errcode))

    def return_ok(self):
        '''Returns zero
        '''
        self.builder.ret(lc.Constant.null(self.lfunc.type.pointee.return_type))

    def op(self, inst):
        if getattr(inst, 'bypass', False):
            return

        # insert temporary attribute
        self.imp_context = codegen_context(imp     = self.implib,
                                           builder = self.builder,
                                           raises  = self.raises,
                                           cast    = self.cast,
                                           lineno  = inst.lineno,
                                           flags   = self.flags)

        attr = 'op_%s' % inst.opcode
        func = getattr(self, attr, self.generic_op)
        result = func(inst)

        del self.imp_context
        return result

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

    def op_retvoid(self, inst):
        self.return_ok()

    def op_ret(self, inst):
        val = self.cast(self.valmap[inst.value], inst.value.type, inst.astype)
        self.builder.store(val, self.lfunc.args[-1])
        self.return_ok()

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
        if getattr(inst.callee, 'type', None) is types.user_function_type:
            callee = inst.callee.value
            lmod, lfunc, retty, argtys, exctable = callee._npm_context_
            self.external_modules.add(lmod)
            declfunc = self.lmod.get_or_insert_function(lfunc.type.pointee,
                                                        name=lfunc.name)
            declfunc.linkage = lc.LINKAGE_LINKONCE_ODR
            args = [self.cast(self.valmap[aval], aval.type, atype)
                    for aval, atype in zip(inst.args, argtys)]

            retptr = self.builder.alloca(retty.llvm_as_value())
            errcode = self.builder.call(declfunc, args + [retptr])

            if exctable:
                # callee defines an exception table
                else_blk = cgutils.append_block(self.builder)
                errswt = self.builder.switch(errcode, else_blk, len(exctable))
                for code, exc in exctable.items():
                    caseblk = cgutils.append_block(self.builder)
                    code = self.add_exception(exc)
                    errswt.add_case(caseblk, types.int32.llvm_const(code))
                    with cgutils.goto_block(self.builder, caseblk):
                        self.return_error(code)
            
            return self.builder.load(retptr)
        else:
            with error_context(during="resolving call %s" % inst.defn):
                if (getattr(inst.defn, 'codegen', None) and
                        hasattr(inst.defn, 'return_type')):
                    # for macros
                    imp = inst.defn
                    args = inst.args
                else:
                    imp = self.implib.get(inst.defn)
                    args = [(self.valmap[aval]
                                if callable(atype)
                                else self.cast(self.valmap[aval], aval.type,
                                               atype))
                            for aval, atype in zip(inst.args, imp.args)]

                assert not inst.kws
                argtys = [aval.type for aval in inst.args]
                return imp(self.imp_context, args, argtys, inst.defn.return_type)

    def op_const(self, inst):
        if isinstance(inst.type.desc, types.BuiltinObject):
            return  # XXX: do not handle builtin object
        return inst.type.llvm_const(inst.value)

    def op_global(self, inst):
        if inst.type is types.function_type:
            return  # do nothing
        elif inst.type is types.user_function_type:
            return  # do nothing
        elif inst.type is types.exception_type:
            return  # do nothing
        elif isinstance(inst.value, macro.Macro):
            if not inst.value.callable:
                imp = self.implib.lookup(inst.value.func, ())
                return imp(self.imp_context, ())
            else:
                return  # do nothing
        else:
            assert False, inst.type

    def op_alias(self, inst):
        return self.valmap[inst.to]

    def op_phi(self, inst):
        values = inst.phi.values()
        if len(values) == 1:
            return self.valmap[values[0]]
        assert False

    def op_tuple(self, inst):
        values = [self.valmap[i] for i in inst.args]
        return inst.type.desc.llvm_pack(self.builder, values)

    def op_slice(self, inst):
        def _get_and_cast(var):
            val = self.valmap[var]
            return self.cast(val, var.type, types.intp)
        start = (_get_and_cast(inst.start)
                    if inst.start is not None
                    else types.intp.llvm_const(0))
        stop = (_get_and_cast(inst.stop)
                    if inst.stop is not None
                    else types.intp.desc.maximum_value())

        if inst.type is types.slice2:
            return inst.type.desc.llvm_pack(self.builder, (start, stop))
        else:
            step = _get_and_cast(inst.start)
            return inst.type.desc.llvm_pack(self.builder, (start, stop, step))

    def op_unpack(self, inst):
        val = self.valmap[inst.value]
        return inst.value.type.llvm_unpack(self.builder, val)[inst.index]

    def op_raise(self, inst):
        args = inst.args
        if len(args) > 1:
            raise ValueError('only support one argument raise statement')
        (excobj,) = args
        if excobj.type is not types.exception_type:
            raise TypeError('can only raise instance of exception')
        self.raises(excobj.value)

