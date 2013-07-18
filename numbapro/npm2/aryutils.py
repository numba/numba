from llvm import core as lc
from . import types

def const_intp(x):
    return types.intp.llvm_const(x)

def auto_intp(x):
    if isinstance(x, int):
        return const_intp(x)
    else:
        return x

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

def preload_attr(builder, ary):
    ndim = ary.type.elements[1].count
    data = builder.extract_value(ary, 0)
    shapeary = builder.extract_value(ary, 1)
    strideary = builder.extract_value(ary, 2)
    shape = [builder.extract_value(shapeary, i) for i in range(ndim)]
    strides = [builder.extract_value(strideary, i) for i in range(ndim)]
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

