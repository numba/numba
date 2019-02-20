"""
Implementation of tuple objects
"""

from llvmlite import ir
import llvmlite.llvmpy.core as lc
import operator

from .imputils import (lower_builtin, lower_getattr_generic, lower_cast,
                       lower_constant,
                       iternext_impl, impl_ret_borrowed, impl_ret_untracked)
from .. import typing, types, cgutils
from ..extending import overload_method


@lower_builtin(types.NamedTupleClass, types.VarArg(types.Any))
def namedtuple_constructor(context, builder, sig, args):
    # A namedtuple has the same representation as a regular tuple
    res = context.make_tuple(builder, sig.return_type, args)
    # The tuple's contents are borrowed
    return impl_ret_borrowed(context, builder, sig.return_type, res)

@lower_builtin(operator.add, types.BaseTuple, types.BaseTuple)
def tuple_add(context, builder, sig, args):
    left, right = [cgutils.unpack_tuple(builder, x) for x in args]
    res = context.make_tuple(builder, sig.return_type, left + right)
    # The tuple's contents are borrowed
    return impl_ret_borrowed(context, builder, sig.return_type, res)

def tuple_cmp_ordered(context, builder, op, sig, args):
    tu, tv = sig.args
    u, v = args
    res = cgutils.alloca_once_value(builder, cgutils.true_bit)
    bbend = builder.append_basic_block("cmp_end")
    for i, (ta, tb) in enumerate(zip(tu.types, tv.types)):
        a = builder.extract_value(u, i)
        b = builder.extract_value(v, i)
        not_equal = context.generic_compare(builder, operator.ne, (ta, tb), (a, b))
        with builder.if_then(not_equal):
            pred = context.generic_compare(builder, op, (ta, tb), (a, b))
            builder.store(pred, res)
            builder.branch(bbend)
    # Everything matched equal => compare lengths
    len_compare = op(len(tu.types), len(tv.types))
    pred = context.get_constant(types.boolean, len_compare)
    builder.store(pred, res)
    builder.branch(bbend)
    builder.position_at_end(bbend)
    return builder.load(res)

@lower_builtin(operator.eq, types.BaseTuple, types.BaseTuple)
def tuple_eq(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    if len(tu.types) != len(tv.types):
        res = context.get_constant(types.boolean, False)
        return impl_ret_untracked(context, builder, sig.return_type, res)
    res = context.get_constant(types.boolean, True)
    for i, (ta, tb) in enumerate(zip(tu.types, tv.types)):
        a = builder.extract_value(u, i)
        b = builder.extract_value(v, i)
        pred = context.generic_compare(builder, operator.eq, (ta, tb), (a, b))
        res = builder.and_(res, pred)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.ne, types.BaseTuple, types.BaseTuple)
def tuple_ne(context, builder, sig, args):
    res = builder.not_(tuple_eq(context, builder, sig, args))
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.lt, types.BaseTuple, types.BaseTuple)
def tuple_lt(context, builder, sig, args):
    res = tuple_cmp_ordered(context, builder, operator.lt, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.le, types.BaseTuple, types.BaseTuple)
def tuple_le(context, builder, sig, args):
    res = tuple_cmp_ordered(context, builder, operator.le, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.gt, types.BaseTuple, types.BaseTuple)
def tuple_gt(context, builder, sig, args):
    res = tuple_cmp_ordered(context, builder, operator.gt, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

@lower_builtin(operator.ge, types.BaseTuple, types.BaseTuple)
def tuple_ge(context, builder, sig, args):
    res = tuple_cmp_ordered(context, builder, operator.ge, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)

# for hashing see hashing.py

@lower_getattr_generic(types.BaseNamedTuple)
def namedtuple_getattr(context, builder, typ, value, attr):
    """
    Fetch a namedtuple's field.
    """
    index = typ.fields.index(attr)
    res = builder.extract_value(value, index)
    return impl_ret_borrowed(context, builder, typ[index], res)


@lower_constant(types.UniTuple)
@lower_constant(types.NamedUniTuple)
def unituple_constant(context, builder, ty, pyval):
    """
    Create a homogeneous tuple constant.
    """
    consts = [context.get_constant_generic(builder, ty.dtype, v)
              for v in pyval]
    return impl_ret_borrowed(
        context, builder, ty, cgutils.pack_array(builder, consts),
    )

@lower_constant(types.Tuple)
@lower_constant(types.NamedTuple)
def unituple_constant(context, builder, ty, pyval):
    """
    Create a heterogeneous tuple constant.
    """
    consts = [context.get_constant_generic(builder, ty.types[i], v)
              for i, v in enumerate(pyval)]
    return impl_ret_borrowed(
        context, builder, ty, cgutils.pack_struct(builder, consts),
    )


#------------------------------------------------------------------------------
# Tuple iterators

@lower_builtin('getiter', types.UniTuple)
@lower_builtin('getiter', types.NamedUniTuple)
def getiter_unituple(context, builder, sig, args):
    [tupty] = sig.args
    [tup] = args

    iterval = context.make_helper(builder, types.UniTupleIter(tupty))

    index0 = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once(builder, index0.type)
    builder.store(index0, indexptr)

    iterval.index = indexptr
    iterval.tuple = tup

    res = iterval._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin('iternext', types.UniTupleIter)
@iternext_impl
def iternext_unituple(context, builder, sig, args, result):
    [tupiterty] = sig.args
    [tupiter] = args

    iterval = context.make_helper(builder, tupiterty, value=tupiter)

    tup = iterval.tuple
    idxptr = iterval.index
    idx = builder.load(idxptr)
    count = context.get_constant(types.intp, tupiterty.container.count)

    is_valid = builder.icmp(lc.ICMP_SLT, idx, count)
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        getitem_sig = typing.signature(tupiterty.container.dtype,
                                       tupiterty.container,
                                       types.intp)
        getitem_out = getitem_unituple(context, builder, getitem_sig,
                                       [tup, idx])
        # As a iternext_impl function, this will incref the yieled value.
        # We need to release the new reference from getitem_unituple.
        if context.enable_nrt:
            context.nrt.decref(builder, tupiterty.container.dtype, getitem_out)
        result.yield_(getitem_out)
        nidx = builder.add(idx, context.get_constant(types.intp, 1))
        builder.store(nidx, iterval.index)


@lower_builtin(operator.getitem, types.UniTuple, types.intp)
@lower_builtin(operator.getitem, types.UniTuple, types.uintp)
@lower_builtin(operator.getitem, types.NamedUniTuple, types.intp)
@lower_builtin(operator.getitem, types.NamedUniTuple, types.uintp)
def getitem_unituple(context, builder, sig, args):
    tupty, _ = sig.args
    tup, idx = args

    errmsg_oob = ("tuple index out of range",)

    if len(tupty) == 0:
        # Empty tuple.

        # Always branch and raise IndexError
        with builder.if_then(cgutils.true_bit):
            context.call_conv.return_user_exc(builder, IndexError,
                                              errmsg_oob)
        # This is unreachable in runtime,
        # but it exists to not terminate the current basicblock.
        res = context.get_constant_null(sig.return_type)
        return impl_ret_untracked(context, builder,
                                  sig.return_type, res)
    else:
        # The tuple is not empty
        bbelse = builder.append_basic_block("switch.else")
        bbend = builder.append_basic_block("switch.end")
        switch = builder.switch(idx, bbelse)

        with builder.goto_block(bbelse):
            context.call_conv.return_user_exc(builder, IndexError,
                                              errmsg_oob)

        lrtty = context.get_value_type(tupty.dtype)
        with builder.goto_block(bbend):
            phinode = builder.phi(lrtty)

        for i in range(tupty.count):
            ki = context.get_constant(types.intp, i)
            bbi = builder.append_basic_block("switch.%d" % i)
            switch.add_case(ki, bbi)
            # handle negative indexing, create case (-tuple.count + i) to
            # reference same block as i
            kin = context.get_constant(types.intp, -tupty.count + i)
            switch.add_case(kin, bbi)
            with builder.goto_block(bbi):
                value = builder.extract_value(tup, i)
                builder.branch(bbend)
                phinode.add_incoming(value, bbi)

        builder.position_at_end(bbend)
        res = phinode
        assert sig.return_type == tupty.dtype
        return impl_ret_borrowed(context, builder, sig.return_type, res)


@lower_builtin('static_getitem', types.BaseTuple, types.IntegerLiteral)
@lower_builtin('static_getitem', types.BaseTuple, types.SliceLiteral)
def static_getitem_tuple(context, builder, sig, args):
    tupty, _ = sig.args
    tup, idx = args
    if isinstance(idx, int):
        if idx < 0:
            idx += len(tupty)
        if not 0 <= idx < len(tupty):
            raise IndexError("cannot index at %d in %s" % (idx, tupty))
        res = builder.extract_value(tup, idx)
    elif isinstance(idx, slice):
        items = cgutils.unpack_tuple(builder, tup)[idx]
        res = context.make_tuple(builder, sig.return_type, items)
    else:
        raise NotImplementedError("unexpected index %r for %s"
                                  % (idx, sig.args[0]))
    return impl_ret_borrowed(context, builder, sig.return_type, res)


#------------------------------------------------------------------------------
# Implicit conversion

@lower_cast(types.BaseTuple, types.BaseTuple)
def tuple_to_tuple(context, builder, fromty, toty, val):
    if (isinstance(fromty, types.BaseNamedTuple)
        or isinstance(toty, types.BaseNamedTuple)):
        # Disallowed by typing layer
        raise NotImplementedError

    if len(fromty) != len(toty):
        # Disallowed by typing layer
        raise NotImplementedError

    olditems = cgutils.unpack_tuple(builder, val, len(fromty))
    items = [context.cast(builder, v, f, t)
             for v, f, t in zip(olditems, fromty, toty)]
    return context.make_tuple(builder, toty, items)


#------------------------------------------------------------------------------
# Methods

@overload_method(types.BaseTuple, 'index')
def tuple_index(tup, value):

    def tuple_index_impl(tup, value):
        for i in range(len(tup)):
            if tup[i] == value:
                return i
        raise ValueError("tuple.index(x): x not in tuple")

    return tuple_index_impl
