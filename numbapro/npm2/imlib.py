from contextlib import contextmanager
import operator
import llvm.core as lc
from . import typesets, types

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

def imp_add_float(builder, args):
    a, b = args
    return builder.fadd(a, b)

def imp_add_complex(dtype):
    def imp(builder, args):
        a, b = args

        a_real, a_imag = dtype.desc.llvm_unpack(builder, a)
        b_real, b_imag = dtype.desc.llvm_unpack(builder, b)

        c_real = imp_add_float(builder, (a_real, b_real))
        c_imag = imp_add_float(builder, (a_imag, b_imag))

        return dtype.desc.llvm_pack(builder, c_real, c_imag)
    return imp


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
    with goto_entry_block(builder):
        # allocate at the beginning
        # assuming a range object must be used statically
        ptr = builder.alloca(obj.type)
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

    # binary add
    imps += binary_op_imp(operator.add, imp_add_integer, typesets.integer_set)
    imps += binary_op_imp(operator.add, imp_add_float, typesets.float_set)
    imps += binary_op_imp(operator.add, imp_add_complex(types.complex64),
                          [types.complex64])
    imps += binary_op_imp(operator.add, imp_add_complex(types.complex128),
                          [types.complex128])

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


#------------------------------------------------------------------------------
# utils

@contextmanager
def goto_entry_block(builder):
    old = builder.basic_block
    entry = builder.basic_block.function.basic_blocks[0]
    builder.position_at_beginning(entry)
    yield
    builder.position_at_end(old)
