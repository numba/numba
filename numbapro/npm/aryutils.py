from contextlib import contextmanager
import llvm.core as lc
from .cgutils import make_array, explode_array, if_then
from .looputils import loop_nest
from .types import const_intp, auto_intp
from . import types

def gep(builder, ptr, indices):
    return builder.gep(ptr, [auto_intp(i) for i in indices])

def setitem(builder, ary, order, indices, value):
    data, shape, strides = preload_attr(builder, ary)
    ptr = getpointer(builder, data, shape, strides, order, indices)
    builder.store(value, ptr)

def getitem(builder, ary, order, indices):
    data, shape, strides = preload_attr(builder, ary)
    ptr = getpointer(builder, data, shape, strides, order, indices)
    val = builder.load(ptr)
    return val

def getndim(ary):
    return ary.type.elements[1].count

def getshape(builder, ary):
    shapeary = builder.extract_value(ary, 1)
    shape = explode_array(builder, shapeary)
    return shape

def getstrides(builder, ary):
    strideary = builder.extract_value(ary, 2)
    strides = explode_array(builder, strideary)
    return strides

def getdata(builder, ary):
    return builder.extract_value(ary, 0)

def preload_attr(builder, ary):
    data = getdata(builder, ary)
    shape = getshape(builder, ary)
    strides = getstrides(builder, ary)
    return data, shape, strides

def getpointer(builder, data, shape, strides, order, indices):
    assert order[0] in 'CFA', "invalid array order code '%s'" % order
    intp = shape[0].type
    if order in 'CF':
        # optimize for C and F contiguous
        steps = []
        if order == 'C':
            for i in range(len(shape)):
                last = const_intp(1)
                for j in shape[i + 1:]:
                    last = builder.mul(last, j)
                steps.append(last)
        elif order =='F':
            for i in range(len(shape)):
                last = const_intp(1)
                for j in shape[:i]:
                    last = builder.mul(last, j)
                steps.append(last)
        else:
            assert False, "unreachable"
        loc = const_intp(0)
        for i, s in zip(indices, steps):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        ptr = builder.gep(data, [loc])
    else:
        # any order
        loc = const_intp(0)
        for i, s in zip(indices, strides):
            tmp = builder.mul(i, s)
            loc = builder.add(loc, tmp)
        base = builder.ptrtoint(data, intp)
        target = builder.add(base, loc)
        ptr = builder.inttoptr(target, data.type)
    return ptr

@contextmanager
def elementwise(builder, ary, aryty):
    '''Context manager for elementwise loop over an array.
    '''
    ndim = aryty.desc.ndim
    begins = (0,) * ndim
    ends = getshape(builder, ary)
    steps = (1,) * ndim

    with loop_nest(builder, begins, ends, steps) as indices:
        val = getitem(builder, ary, indices=indices, order=aryty.desc.order)
        yield val

def make_intp_array(builder, values):
    llintp = types.intp.llvm_as_value()
    return make_array(builder, llintp, [auto_intp(x) for x in values])

def ndarray(builder, dtype, ndim, order, data, shape, strides):
    shape = make_intp_array(builder, shape)
    strides = make_intp_array(builder, strides)
    aryty = types.arraytype(dtype, ndim, order)
    return aryty.desc.llvm_pack(builder, (data, shape, strides))

def view(builder, ary, begins, ends, order):
    assert order[0] in 'CFA'
    if order[0] not in 'CF':
        raise TypeError('array must be inner contiguous; order=%s' % order)

    ndim = getndim(ary)
    assert ndim == len(begins)
    assert ndim == len(ends)

    data = getdata(builder, ary)
    shape = getshape(builder, ary)
    strides = getstrides(builder, ary)

    newdata = getpointer(builder, data, shape, strides, order, begins)
    dimlens = [builder.sub(auto_intp(e), auto_intp(b))
               for b, e in zip(begins, ends)]
    newshape = make_intp_array(builder, dimlens)

    newary = builder.insert_value(ary, newdata, 0)
    newary = builder.insert_value(newary, newshape, 1)
    return newary

def boundcheck(builder, raises, ary, indices):
    assert getndim(ary) == len(indices), 'index dimension mismatch'
    shape = getshape(builder, ary)
    for i, s in zip(indices, shape):
        i = auto_intp(i)
        oob = builder.icmp(lc.ICMP_UGE, i, s)
        with if_then(builder, oob):
            raises(IndexError, 'out of bound')

def wraparound(builder, ary, indices):
    assert getndim(ary) == len(indices), 'index dimension mismatch'
    shape = getshape(builder, ary)
    return [axis_wraparound(builder, i, s)
            for i, s in zip(indices, shape)]

def axis_wraparound(builder, val, end):
    '''
    :param val: [int|llvm Value of intp] the value to be wrapped around.
    :param end: [llvm Value of intp] the max+1 value for val.
    '''
    val = auto_intp(val)
    neg = builder.icmp(lc.ICMP_SLT, val, types.intp.llvm_const(0))
    normed = builder.select(neg, builder.add(val, end), val)
    return normed

def clip(builder, ary, indices, plus1=False):
    '''
    :param val: [int|llvm Value of intp] the value
    '''
    assert getndim(ary) == len(indices), 'index dimension mismatch'
    shape = getshape(builder, ary)
    if plus1:
        return [axis_clip(builder, i, s)
                for i, s in zip(indices, shape)]
    else:
        return [axis_clip(builder, i, builder.sub(s, auto_intp(1)))
                for i, s in zip(indices, shape)]

def axis_clip(builder, val, bound):
    '''
    :param val: [int|llvm Value of intp] the value to be clipped.
    :param bound: [llvm Value of intp] the upperbound
    '''
    val = auto_intp(val)
    clipping = builder.icmp(lc.ICMP_UGT, val, bound)
    res = builder.select(clipping, bound, val)
    return res
