from contextlib import contextmanager
from llvm import core as lc
from . import types
from .cgutils import const_intp, auto_intp, loop_nest

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

def getndim(builder, ary):
    return ary.type.elements[1].count

def getshape(builder, ary):
    ndim = getndim(builder, ary)
    shapeary = builder.extract_value(ary, 1)
    shape = [builder.extract_value(shapeary, i) for i in range(ndim)]
    return shape

def getstrides(builder, ary):
    ndim = getndim(builder, ary)
    strideary = builder.extract_value(ary, 2)
    strides = [builder.extract_value(strideary, i) for i in range(ndim)]
    return strides

def getdata(builder, ary):
    return builder.extract_value(ary, 0)

def preload_attr(builder, ary):
    ndim = getndim(builder, ary)
    data = getdata(builder, ary)
    shape = getshape(builder, ary)
    strides = getstrides(builder, ary)
    return data, shape, strides

def getpointer(builder, data, shape, strides, order, indices):
    assert order in 'CFA', "invalid array order code '%s'" % order
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
