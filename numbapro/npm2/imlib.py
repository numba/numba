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

# binary add

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

# binary sub

def imp_sub_integer(builder, args):
    a, b = args
    return builder.sub(a, b)

def imp_sub_float(builder, args):
    a, b = args
    return builder.fsub(a, b)

def imp_sub_complex(dtype):
    def imp(builder, args):
        a, b = args

        a_real, a_imag = dtype.desc.llvm_unpack(builder, a)
        b_real, b_imag = dtype.desc.llvm_unpack(builder, b)

        c_real = imp_sub_float(builder, (a_real, b_real))
        c_imag = imp_sub_float(builder, (a_imag, b_imag))

        return dtype.desc.llvm_pack(builder, c_real, c_imag)
    return imp


# binary mul

def imp_mul_integer(builder, args):
    a, b = args
    return builder.mul(a, b)

def imp_mul_float(builder, args):
    a, b = args
    return builder.fmul(a, b)

def imp_mul_complex(dtype):
    '''
    x y = (a c - b d) + i (a d + b c)
    '''
    def imp(builder, args):
        x, y = args

        a, b = dtype.desc.llvm_unpack(builder, x)
        c, d = dtype.desc.llvm_unpack(builder, y)

        ac = imp_mul_float(builder, (a, c))
        bd = imp_mul_float(builder, (b, d))
        ad = imp_mul_float(builder, (a, d))
        bc = imp_mul_float(builder, (b, c))

        real = imp_sub_float(builder, (ac, bd))
        imag = imp_add_float(builder, (ad, bc))

        return dtype.desc.llvm_pack(builder, real, imag)
    return imp

# binary floordiv

def imp_floordiv_signed(builder, args):
    a, b = args
    return builder.sdiv(a, b)

def imp_floordiv_unsigned(builder, args):
    a, b = args
    return builder.udiv(a, b)

def imp_floordiv_float(intty):
    def imp(builder, args):
        a, b = args
        return builder.fptosi(builder.fdiv(a, b),
                              lc.Type.int(intty.desc.bitwidth))
    return imp

# binary truediv

def imp_truediv_float(builder, args):
    a, b = args
    return builder.fdiv(a, b)

# binary mod

def imp_mod_signed(builder, args):
    a, b = args
    return builder.srem(a, b)

def imp_mod_unsigned(builder, args):
    a, b = args
    return builder.urem(a, b)

def imp_mod_float(builder, args):
    a, b = args
    return builder.frem(a, b)


# range

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

#----------------------------------------------------------------------------
# utils

def bool_op_imp(funcobj, imp, typeset):
    return [Imp(imp, funcobj, args=(ty, ty), return_type=types.boolean)
            for ty in typeset]

def binary_op_imp(funcobj, imp, typeset):
    return [Imp(imp, funcobj, args=(ty, ty), return_type=ty)
            for ty in typeset]

def floordiv_imp(funcobj, imp, ty, ret):
    return [Imp(imp(ret), funcobj, args=(ty, ty), return_type=ret)]

def populate_builtin_impl(implib):
    imps = []

    # binary add
    imps += binary_op_imp(operator.add, imp_add_integer, typesets.integer_set)
    imps += binary_op_imp(operator.add, imp_add_float, typesets.float_set)
    imps += binary_op_imp(operator.add, imp_add_complex(types.complex64),
                          [types.complex64])
    imps += binary_op_imp(operator.add, imp_add_complex(types.complex128),
                          [types.complex128])

    # binary sub
    imps += binary_op_imp(operator.sub, imp_sub_integer, typesets.integer_set)
    imps += binary_op_imp(operator.sub, imp_sub_float, typesets.float_set)
    imps += binary_op_imp(operator.sub, imp_sub_complex(types.complex64),
                          [types.complex64])
    imps += binary_op_imp(operator.sub, imp_sub_complex(types.complex128),
                          [types.complex128])
    
    # binary mul
    imps += binary_op_imp(operator.mul, imp_mul_integer, typesets.integer_set)
    imps += binary_op_imp(operator.mul, imp_mul_float, typesets.float_set)
    imps += binary_op_imp(operator.mul, imp_mul_complex(types.complex64),
                          [types.complex64])
    imps += binary_op_imp(operator.mul, imp_mul_complex(types.complex128),
                          [types.complex128])

    # binary floordiv
    imps += binary_op_imp(operator.floordiv, imp_floordiv_signed,
                          typesets.signed_set)
    imps += binary_op_imp(operator.floordiv, imp_floordiv_unsigned,
                          typesets.unsigned_set)
    imps += floordiv_imp(operator.floordiv, imp_floordiv_float,
                         types.float32, types.int32)
    imps += floordiv_imp(operator.floordiv, imp_floordiv_float,
                         types.float64, types.int64)

    # binary truediv
    imps += binary_op_imp(operator.truediv, imp_truediv_float,
                          typesets.float_set)

    # binary mod
    imps += binary_op_imp(operator.mod, imp_mod_signed, typesets.signed_set)
    imps += binary_op_imp(operator.mod, imp_mod_unsigned, typesets.unsigned_set)
    imps += binary_op_imp(operator.mod, imp_mod_float, typesets.float_set)

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
