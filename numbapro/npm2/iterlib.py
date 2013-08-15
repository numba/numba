import llvm.core as lc
from . import types, cgutils

def match_range_iter(args):
    it, = args
    if it == types.range_iter_type:
        return it

def enumerate_return(args):
    inner, = args
    return types.enumerate_type(inner.desc.iterator())

def enumerate_iter_return(args):
    it, = args
    return types.tupletype(types.intp,
                           it.desc.state.desc.inner.desc.iterate_data())

def iter_enumerate_return(args):
    enum, = args
    return types.enumerate_iter_type(enum)

def make_range_obj(context, start, stop, step):
    rangetype = types.range_type.llvm_as_value()
    rangeobj = lc.Constant.undef(rangetype)
    
    rangeobj = context.builder.insert_value(rangeobj, start, 0)
    rangeobj = context.builder.insert_value(rangeobj, stop, 1)
    rangeobj = context.builder.insert_value(rangeobj, step, 2)
    return rangeobj

class Range1(object):
    function = range, (types.intp,), types.range_type

    def generic_implement(self, context, args, argtys, retty):
        assert len(args) == 1
        (stop,) = args
        start = types.intp.llvm_const(0)
        step = types.intp.llvm_const(1)
        return make_range_obj(context, start, stop, step)

class Range2(object):
    function = range, (types.intp, types.intp), types.range_type

    def generic_implement(self, context, args, argtys, retty):
        assert len(args) == 2
        start, stop = args
        step = types.intp.llvm_const(1)
        return make_range_obj(context, start, stop, step)

class Range3(object):
    function = range, (types.intp, types.intp, types.intp), types.range_type

    def generic_implement(self, context, args, argtys, retty):
        assert len(args) == 3
        start, stop, step = args
        return make_range_obj(context, start, stop, step)

class XRange1(Range1):
    function = xrange, (types.intp,), types.range_type

class XRange2(Range2):
    function = xrange, (types.intp, types.intp), types.range_type

class XRange3(Range3):
    function = xrange, (types.intp, types.intp, types.intp), types.range_type

class IterRange(object):
    function = iter, (types.range_type,), types.range_iter_type

    def generic_implement(self, context, args, argtys, retty):
        obj, = args
        with cgutils.goto_entry_block(context.builder):
            # allocate at the beginning
            # assuming a range object must be used statically
            ptr = context.builder.alloca(types.range_iter_type.llvm_as_value().pointee)
        context.builder.store(obj, ptr)
        return ptr

class RangeIterValid(object):
    function = ('itervalid',
                (match_range_iter,),
                types.boolean)

    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ptr, = args
        idx0 = types.int32.llvm_const(0)
        idx1 = types.int32.llvm_const(1)
        idx2 = types.int32.llvm_const(2)
        start = builder.load(builder.gep(ptr, [idx0, idx0]))
        stop = builder.load(builder.gep(ptr, [idx0, idx1]))
        step = builder.load(builder.gep(ptr, [idx0, idx2]))

        zero = types.intp.llvm_const(0)
        positive = builder.icmp(lc.ICMP_SGE, step, zero)
        posok = builder.icmp(lc.ICMP_SLT, start, stop)
        negok = builder.icmp(lc.ICMP_SGT, start, stop)
        
        return builder.select(positive, posok, negok)

class RangeIterNext(object):
    function = ('iternext',
                (match_range_iter,),
                types.intp)
    
    def generic_implement(self, context, args, argtys, retty):
        builder = context.builder
        ptr, = args
        idx0 = types.int32.llvm_const(0)
        idx2 = types.int32.llvm_const(2)
        startptr = builder.gep(ptr, [idx0, idx0])
        start = builder.load(startptr)
        step = builder.load(builder.gep(ptr, [idx0, idx2]))
        next = builder.add(start, step)
        builder.store(next, startptr)
        return start

class Enumerate(object):
    function = enumerate, (types.IteratorFactoryKind,), enumerate_return

    def generic_implement(self, context, args, argtys, retty):
        (iterfacttype,) = argtys
        iterimptype = iterfacttype.desc.imp
        (iterfact,) = args

        builder = context.builder
        enumtype = retty.llvm_as_value()
        enumobj = lc.Constant.undef(enumtype)
        enumobj = builder.insert_value(enumobj, types.const_intp(0), 0)
        with cgutils.goto_entry_block(builder):
            # allocate iterator 
            iterptr = builder.alloca(iterfact.type)
        enumobj = builder.insert_value(enumobj, iterptr, 1)
        builder.store(iterfact, iterptr)
        return enumobj

class IterEnumerate(object):
    function = iter, (types.EnumerateKind,), iter_enumerate_return

    def generic_implement(self, context, args, argtys, retty):
        iterval, = args
        builder = context.builder
        with cgutils.goto_entry_block(builder):
            # allocate iterator
            iterptr = builder.alloca(iterval.type)
        builder.store(iterval, iterptr)
        return iterptr

class EnumerateIterValid(object):
    function = 'itervalid', (types.EnumerateIterKind,), types.boolean

    def generic_implement(self, context, args, argtys, retty):
        (enumitertype,) = argtys
        (enumiter,) = args
        builder = context.builder
        inneritertype = enumitertype.desc.inner()
        inneriter = builder.extract_value(builder.load(enumiter), 1)
        innervalid = context.imp.lookup('itervalid', (inneritertype,))
        valid = innervalid(context, (inneriter,))
        return valid

class EnumerateIterNext(object):
    function = 'iternext', (types.EnumerateIterKind,), enumerate_iter_return

    def generic_implement(self, context, args, argtys, retty):
        (enumitertype,) = argtys
        (enumiter,) = args
        builder = context.builder
        # invoke inner iterator next
        inneritertype = enumitertype.desc.inner()
        inneriter = builder.extract_value(builder.load(enumiter), 1)
        innernext = context.imp.lookup('iternext', (inneritertype,))
        innerout = innernext(context, (inneriter,))
        # increment enumerator
        counterptr = builder.gep(enumiter, [types.const_intp(0),
                                            types.int32.llvm_const(0)],
                                 inbounds=True)
        counter = builder.load(counterptr)
        ncounter = builder.add(counter, types.const_intp(1))
        builder.store(ncounter, counterptr)
        return retty.desc.llvm_pack(builder, (counter, innerout))


extensions = [
    # range/xrange
    Range1, XRange1,
    Range2, XRange2,
    Range3, XRange3,
    IterRange,
    RangeIterValid,
    RangeIterNext,
    # enumerate
    Enumerate,
    IterEnumerate,
    EnumerateIterValid,
    EnumerateIterNext,
]