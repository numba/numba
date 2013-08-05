from contextlib import contextmanager
from .cgutils import (const_intp, auto_intp, loop_nest, make_array,
                      explode_array,)
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
    ndim = getndim(ary)
    shapeary = builder.extract_value(ary, 1)
    shape = explode_array(builder, shapeary)
    return shape

def getstrides(builder, ary):
    ndim = getndim(ary)
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

def view(builder, ary, begins, ends, order):
    assert order[0] in 'CFA'
    if order[0] not in 'CF':
        raise TypeError('array must be inner contiguous; order=%s' % order)

    if order[0] == 'C':
        innermost = -1
        outermost = 0
    elif order[0] == 'F':
        innermost = 0
        outermost = -1
    else:
        assert 'unreachable'

    ndim = getndim(ary)
    assert ndim == len(begins)
    assert ndim == len(ends)

    data = getdata(builder, ary)

    if ndim == 1:
        offset = auto_intp(begins[innermost])
        newdata = builder.gep(data, [offset], inbounds=True)
        dimlens = [builder.sub(auto_intp(e), auto_intp(b))
                   for b, e in zip(begins, ends)]
        newshape = make_intp_array(builder, dimlens)

        newary = builder.insert_value(ary, newdata, 0)
        newary = builder.insert_value(newary, newshape, 1)

    return newary



