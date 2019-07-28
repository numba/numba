"""
Compiler-side implementation of the Numba  typed-list.
"""
import ctypes
import operator
from enum import IntEnum

from llvmlite import ir

from numba import cgutils
from numba import _helperlib

from numba.extending import (
    overload,
    overload_method,
    register_jitable,
    intrinsic,
    register_model,
    models,
    lower_builtin,
)
from numba.targets.imputils import iternext_impl
from numba import types
from numba.types import (
    ListType,
    ListTypeIterableType,
    ListTypeIteratorType,
    Type,
)
from numba.targets.imputils import impl_ret_borrowed, RefType
from numba.errors import TypingError
from numba import typing
from numba.typedobjectutils import (_as_bytes,
                                    _cast,
                                    _nonoptional,
                                    _get_incref_decref,
                                    _container_get_data,
                                    _container_get_meminfo,
                                    )


ll_list_type = cgutils.voidptr_t
ll_listiter_type = cgutils.voidptr_t
ll_voidptr_type = cgutils.voidptr_t
ll_status = cgutils.int32_t
ll_ssize_t = cgutils.intp_t
ll_bytes = cgutils.voidptr_t


_meminfo_listptr = types.MemInfoPointer(types.voidptr)

INDEXTY = types.intp


@register_model(ListType)
class ListModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('meminfo', _meminfo_listptr),
            ('data', types.voidptr),   # ptr to the C list
        ]
        super(ListModel, self).__init__(dmm, fe_type, members)


@register_model(ListTypeIterableType)
@register_model(ListTypeIteratorType)
class ListIterModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('parent', fe_type.parent),  # reference to the list
            ('state', types.voidptr),    # iterator state in C code
        ]
        super(ListIterModel, self).__init__(dmm, fe_type, members)


class ListStatus(IntEnum):
    """Status code for other list operations.
    """
    LIST_OK = 0,
    LIST_ERR_INDEX = -1
    LIST_ERR_NO_MEMORY = -2
    LIST_ERR_MUTATED = -3
    LIST_ERR_ITER_EXHAUSTED = -4


def _raise_if_error(context, builder, status, msg):
    """Raise an internal error depending on the value of *status*
    """
    ok_status = status.type(int(ListStatus.LIST_OK))
    with builder.if_then(builder.icmp_signed('!=', status, ok_status),
                         likely=True):
        context.call_conv.return_user_exc(builder, RuntimeError, (msg,))


@intrinsic
def _as_meminfo(typingctx, lstobj):
    """Returns the MemInfoPointer of a list.
    """
    if not isinstance(lstobj, types.ListType):
        raise TypingError('expected *lstobj* to be a ListType')

    def codegen(context, builder, sig, args):
        [tl] = sig.args
        [l] = args
        # Incref
        context.nrt.incref(builder, tl, l)
        ctor = cgutils.create_struct_proxy(tl)
        lstruct = ctor(context, builder, value=l)
        # Returns the plain MemInfo
        return lstruct.meminfo

    sig = _meminfo_listptr(lstobj)
    return sig, codegen


@intrinsic
def _from_meminfo(typingctx, mi, listtyperef):
    """Recreate a list from a MemInfoPointer
    """
    if mi != _meminfo_listptr:
        raise TypingError('expected a MemInfoPointer for list.')
    listtype = listtyperef.instance_type
    if not isinstance(listtype, ListType):
        raise TypingError('expected a {}'.format(ListType))

    def codegen(context, builder, sig, args):
        [tmi, tdref] = sig.args
        td = tdref.instance_type
        [mi, _] = args

        ctor = cgutils.create_struct_proxy(td)
        dstruct = ctor(context, builder)

        data_pointer = context.nrt.meminfo_data(builder, mi)
        data_pointer = builder.bitcast(data_pointer, ll_list_type.as_pointer())

        dstruct.data = builder.load(data_pointer)
        dstruct.meminfo = mi

        return impl_ret_borrowed(
            context,
            builder,
            listtype,
            dstruct._getvalue(),
        )

    sig = listtype(mi, listtyperef)
    return sig, codegen


@intrinsic
def _list_set_method_table(typingctx, lp, itemty):
    """Wrap numba_list_set_method_table
    """
    resty = types.void
    sig = resty(lp, itemty)

    def codegen(context, builder, sig, args):
        vtablety = ir.LiteralStructType([
            ll_voidptr_type,  # item incref
            ll_voidptr_type,  # item decref
        ])
        setmethod_fnty = ir.FunctionType(
            ir.VoidType(),
            [ll_list_type, vtablety.as_pointer()]
        )
        setmethod_fn = ir.Function(
            builder.module,
            setmethod_fnty,
            name='numba_list_set_method_table',
        )
        dp = args[0]
        vtable = cgutils.alloca_once(builder, vtablety, zfill=True)

        # install item incref/decref
        item_incref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 0)
        item_decref_ptr = cgutils.gep_inbounds(builder, vtable, 0, 1)

        dm_item = context.data_model_manager[itemty.instance_type]
        if dm_item.contains_nrt_meminfo():
            item_incref, item_decref = _get_incref_decref(
                context, builder.module, dm_item, "list"
            )
            builder.store(
                builder.bitcast(item_incref, item_incref_ptr.type.pointee),
                item_incref_ptr,
            )
            builder.store(
                builder.bitcast(item_decref, item_decref_ptr.type.pointee),
                item_decref_ptr,
            )

        builder.call(setmethod_fn, [dp, vtable])

    return sig, codegen


@lower_builtin(operator.is_, types.ListType, types.ListType)
def list_is(context, builder, sig, args):
    a_meminfo = _container_get_meminfo(context, builder, sig.args[0], args[0])
    b_meminfo = _container_get_meminfo(context, builder, sig.args[1], args[1])
    ma = builder.ptrtoint(a_meminfo, cgutils.intp_t)
    mb = builder.ptrtoint(b_meminfo, cgutils.intp_t)
    return builder.icmp_signed('==', ma, mb)


def _call_list_free(context, builder, ptr):
    """Call numba_list_free(ptr)
    """
    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_list_type],
    )
    free = builder.module.get_or_insert_function(fnty, name='numba_list_free')
    builder.call(free, [ptr])


# FIXME: this needs a careful review
def _imp_dtor(context, module):
    """Define the dtor for list
    """
    llvoidptr = context.get_value_type(types.voidptr)
    llsize = context.get_value_type(types.uintp)
    fnty = ir.FunctionType(
        ir.VoidType(),
        [llvoidptr, llsize, llvoidptr],
    )
    fname = '_numba_list_dtor'
    fn = module.get_or_insert_function(fnty, name=fname)

    if fn.is_declaration:
        # Set linkage
        fn.linkage = 'linkonce_odr'
        # Define
        builder = ir.IRBuilder(fn.append_basic_block())
        lp = builder.bitcast(fn.args[0], ll_list_type.as_pointer())
        l = builder.load(lp)
        _call_list_free(context, builder, l)
        builder.ret_void()

    return fn


def new_list(item):
    """Construct a new list. (Not implemented in the interpreter yet)

    Parameters
    ----------
    item: TypeRef
        Item type of the new list.
    """
    raise NotImplementedError


@intrinsic
def _make_list(typingctx, itemty, ptr):
    """Make a list struct with the given *ptr*

    Parameters
    ----------
    itemty: Type
        Type of the item.
    ptr : llvm pointer value
        Points to the list object.
    """
    list_ty = types.ListType(itemty.instance_type)

    def codegen(context, builder, signature, args):
        [_, ptr] = args
        ctor = cgutils.create_struct_proxy(list_ty)
        lstruct = ctor(context, builder)
        lstruct.data = ptr

        alloc_size = context.get_abi_sizeof(
            context.get_value_type(types.voidptr),
        )
        dtor = _imp_dtor(context, builder.module)
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder,
            context.get_constant(types.uintp, alloc_size),
            dtor,
        )

        data_pointer = context.nrt.meminfo_data(builder, meminfo)
        data_pointer = builder.bitcast(data_pointer, ll_list_type.as_pointer())
        builder.store(ptr, data_pointer)

        lstruct.meminfo = meminfo

        return lstruct._getvalue()

    sig = list_ty(itemty, ptr)
    return sig, codegen


@intrinsic
def _list_new(typingctx, itemty):
    """Wrap numba_list_new.

    Allocate a new list object with zero capacity.

    Parameters
    ----------
    itemty: Type
        Type of the items

    """
    resty = types.voidptr
    sig = resty(itemty)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type.as_pointer(), ll_ssize_t, ll_ssize_t],
        )
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_new')
        # Determine sizeof item types
        ll_item = context.get_data_type(itemty.instance_type)
        sz_item = context.get_abi_sizeof(ll_item)
        reflp = cgutils.alloca_once(builder, ll_list_type, zfill=True)
        status = builder.call(
            fn,
            [reflp, ll_ssize_t(sz_item), ll_ssize_t(0)],
        )
        _raise_if_error(
            context, builder, status,
            msg="Failed to allocate list",
        )
        lp = builder.load(reflp)
        return lp

    return sig, codegen


@overload(new_list)
def impl_new_list(item):
    """Creates a new list with *item* as the type
    of the list item.
    """
    if not isinstance(item, Type):
        raise TypeError("expecting *item* to be a numba Type")

    itemty = item

    def imp(item):
        lp = _list_new(itemty)
        _list_set_method_table(lp, itemty)
        l = _make_list(itemty, lp)
        return l

    return imp


@overload(len)
def impl_len(l):
    """len(list)
    """
    if isinstance(l, types.ListType):
        def impl(l):
            return _list_length(l)

        return impl


@intrinsic
def _list_length(typingctx, l):
    """Wrap numba_list_length

    Returns the length of the list.
    """
    resty = types.intp
    sig = resty(l)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_ssize_t,
            [ll_list_type],
        )
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_length')
        [l] = args
        [tl] = sig.args
        lp = _container_get_data(context, builder, tl, l)
        n = builder.call(fn, [lp])
        return n

    return sig, codegen


@intrinsic
def _list_append(typingctx, l, item):
    """Wrap numba_list_append
    """
    resty = types.int32
    sig = resty(l, l.item_type)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_bytes],
        )
        [l, item] = args
        [tl, titem] = sig.args
        fn = builder.module.get_or_insert_function(fnty, name='numba_list_append')

        dm_item = context.data_model_manager[titem]

        data_item = dm_item.as_data(builder, item)

        ptr_item = cgutils.alloca_once_value(builder, data_item)

        lp = _container_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                _as_bytes(builder, ptr_item),
            ],
        )
        return status

    return sig, codegen


@overload_method(types.ListType, 'append')
def impl_append(l, item):
    if not isinstance(l, types.ListType):
        return

    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        status = _list_append(l, casteditem)
        if status == ListStatus.LIST_OK:
            return
        elif status == ListStatus.LIST_ERR_NO_MEMORY:
            raise MemoryError('Unable to allocate memory to append item')
        else:
            raise RuntimeError('list.append failed unexpectedly')

    if l.is_precise():
        # Handle the precise case.
        return impl
    else:
        # Handle the imprecise case.
        l = l.refine(item)
        # Re-bind the item type to match the arguments.
        itemty = l.item_type
        # Create the signature that we wanted this impl to have.
        sig = typing.signature(types.void, l, itemty)
        return sig, impl


@register_jitable
def handle_index(l, index):
    """Handle index.

    If the index is negative, convert it. If the index is out of range, raise
    an IndexError.
    """
    # convert negative indices to positive ones
    if index < 0:
        index = len(l) + index
    # check that the index is in range
    if not (0 <= index < len(l)):
        raise IndexError("list index out of range")
    return index


@register_jitable
def handle_slice(l, s):
    """Handle slice.

    Convert a slice object for a given list into a range object that can be
    used to index the list. Many subtle caveats here, especially if the step is
    negative.
    """
    if len(l) == 0:  # ignore slice for empty list
        return range(0)
    ll, sa, so, se = len(l), s.start, s.stop, s.step
    if se > 0:
        start = max(ll + sa,  0) if s.start < 0 else min(ll, sa)
        stop = max(ll + so, 0) if so < 0 else min(ll, so)
    elif se < 0:
        start = max(ll + sa,  -1) if s.start < 0 else min(ll - 1, sa)
        stop = max(ll + so, -1) if so < 0 else min(ll, so)
    else:
        # should be caught earlier, but isn't, so we raise here
        raise ValueError("slice step cannot be zero")
    return range(start, stop, s.step)


@intrinsic
def _list_getitem(typingctx, l, index):
    return _list_getitem_pop_helper(typingctx, l, index, 'getitem')


@intrinsic
def _list_pop(typingctx, l, index):
    return _list_getitem_pop_helper(typingctx, l, index, 'pop')


def _list_getitem_pop_helper(typingctx, l, index, op):
    """Wrap numba_list_getitem and numba_list_pop

    Returns 2-tuple of (intp, ?item_type)

    This is a helper that is parametrized on the type of operation, which can
    be either 'pop' or 'getitem'. This is because, signature wise, getitem and
    pop and are the same.
    """
    assert(op in ("pop", "getitem"))
    resty = types.Tuple([types.int32, types.Optional(l.item_type)])
    sig = resty(l, index)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_ssize_t, ll_bytes],
        )
        [tl, tindex] = sig.args
        [l, index] = args
        fn = builder.module.get_or_insert_function(fnty,
                                                   name='numba_list_{}'.format(op))

        dm_item = context.data_model_manager[tl.item_type]
        ll_item = context.get_data_type(tl.item_type)
        ptr_item = cgutils.alloca_once(builder, ll_item)

        lp = _container_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                index,
                _as_bytes(builder, ptr_item),
            ],
        )
        # Load item if output is available
        found = builder.icmp_signed('>=', status, status.type(int(ListStatus.LIST_OK)))

        out = context.make_optional_none(builder, tl.item_type)
        pout = cgutils.alloca_once_value(builder, out)

        with builder.if_then(found):
            item = dm_item.load_from_data_pointer(builder, ptr_item)
            context.nrt.incref(builder, tl.item_type, item)
            loaded = context.make_optional_value(builder, tl.item_type, item)
            builder.store(loaded, pout)

        out = builder.load(pout)
        return context.make_tuple(builder, resty, [status, out])

    return sig, codegen


@overload(operator.getitem)
def impl_getitem(l, index):
    if not isinstance(l, types.ListType):
        return

    indexty = INDEXTY
    itemty = l.item_type

    if index in types.signed_domain:
        def integer_impl(l, index):
            index = handle_index(l, index)
            castedindex = _cast(index, indexty)
            status, item = _list_getitem(l, castedindex)
            if status == ListStatus.LIST_OK:
                return _nonoptional(item)
            else:
                raise AssertionError("internal list error during getitem")

        return integer_impl

    elif isinstance(index, types.SliceType):
        def slice_impl(l, index):
            newl = new_list(itemty)
            for i in handle_slice(l, index):
                newl.append(l[i])
            return newl

        return slice_impl

    else:
        raise TypingError("list indices must be signed integers or slices")


@intrinsic
def _list_setitem(typingctx, l, index, item):
    """Wrap numba_list_setitem
    """
    resty = types.int32
    sig = resty(l, index, item)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_ssize_t, ll_bytes],
        )
        [l, index, item] = args
        [tl, tindex, titem] = sig.args
        fn = builder.module.get_or_insert_function(fnty,
                                                   name='numba_list_setitem')

        dm_item = context.data_model_manager[titem]
        data_item = dm_item.as_data(builder, item)
        ptr_item = cgutils.alloca_once_value(builder, data_item)

        lp = _container_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                index,
                _as_bytes(builder, ptr_item),
            ],
        )
        return status

    return sig, codegen


@overload(operator.setitem)
def impl_setitem(l, index, item):
    if not isinstance(l, types.ListType):
        return

    indexty = INDEXTY
    itemty = l.item_type

    if index in types.signed_domain:
        def impl_integer(l, index, item):
            index = handle_index(l, index)
            castedindex = _cast(index, indexty)
            casteditem = _cast(item, itemty)
            status = _list_setitem(l, castedindex, casteditem)
            if status == ListStatus.LIST_OK:
                return
            else:
                raise AssertionError("internal list error during settitem")

        return impl_integer

    elif isinstance(index, types.SliceType):
        if not isinstance(item, types.IterableType):
            raise TypingError("can only assign an iterable when using a slice "
                              "with assignment/setitem")

        def impl_slice(l, index, item):
            # special case "a[i:j] = a", need to copy first
            if l == item:
                item = item.copy()
            slice_range = handle_slice(l, index)
            # non-extended (simple) slices
            if slice_range.step == 1:
                # replace
                if len(item) == len(slice_range):
                    for i, j in zip(slice_range, item):
                        l[i] = j
                # replace and insert
                if len(item) > len(slice_range):
                    # do the replaces we can
                    for i, j in zip(slice_range, item[:len(slice_range)]):
                        l[i] = j
                    # insert the remaining ones
                    insert_range = range(slice_range.stop,
                                         slice_range.stop +
                                         len(item) - len(slice_range))
                    for i, k in zip(insert_range, item[len(slice_range):]):
                        # FIXME: This may be slow.  Each insert can incur a
                        # memory copy of one or more items.
                        l.insert(i, k)
                # replace and delete
                if len(item) < len(slice_range):
                    # do the replaces we can
                    replace_range = range(slice_range.start,
                                          slice_range.start + len(item))
                    for i,j in zip(replace_range, item):
                        l[i] = j
                    # delete remaining ones
                    del l[slice_range.start + len(item):slice_range.stop]
            # Extended slices
            else:
                if len(slice_range) != len(item):
                    raise ValueError("length mismatch for extended slice and sequence")
                # extended slice can only replace
                for i, j in zip(slice_range, item):
                    l[i] = j

        return impl_slice

    else:
        raise TypingError("list indices must be signed integers or slices")


@overload_method(types.ListType, 'pop')
def impl_pop(l, index=-1):
    if not isinstance(l, types.ListType):
        return

    indexty = INDEXTY

    # FIXME: this type check works, but it isn't clear why and if it optimal
    if (isinstance(index, int)
            or index in types.signed_domain
            or isinstance(index, types.Omitted)):
        def impl(l, index=-1):
            if len(l) == 0:
                raise IndexError("pop from empty list")
            index = handle_index(l, index)
            castedindex = _cast(index, indexty)
            status, item = _list_pop(l, castedindex)
            if status == ListStatus.LIST_OK:
                return _nonoptional(item)
            else:
                raise AssertionError("internal list error during pop")
        return impl

    else:
        raise TypingError("argument for pop must be a signed integer")


@intrinsic
def _list_delete_slice(typingctx, l, start, stop, step):
    """Wrap numba_list_delete_slice
    """
    resty = types.int32
    sig = resty(l, start, stop, step)

    def codegen(context, builder, sig, args):
        fnty = ir.FunctionType(
            ll_status,
            [ll_list_type, ll_ssize_t, ll_ssize_t, ll_ssize_t],
        )
        [l, start, stop, step] = args
        [tl, tstart, tstop, tstep] = sig.args
        fn = builder.module.get_or_insert_function(fnty,
                                                   name='numba_list_delete_slice')

        lp = _container_get_data(context, builder, tl, l)
        status = builder.call(
            fn,
            [
                lp,
                start,
                stop,
                step,
            ],
        )
        return status

    return sig, codegen


@overload(operator.delitem)
def impl_delitem(l, index):
    if not isinstance(l, types.ListType):
        return

    if index in types.signed_domain:
        def integer_impl(l, index):
            l.pop(index)

        return integer_impl

    elif isinstance(index, types.SliceType):
        def slice_impl(l, index):
            slice_range = handle_slice(l, index)
            _list_delete_slice(l,
                               slice_range.start,
                               slice_range.stop,
                               slice_range.step)
        return slice_impl

    else:
        raise TypingError("list indices must be signed integers or slices")


@overload(operator.contains)
def impl_contains(l, item):
    if not isinstance(l, types.ListType):
        return

    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        for i in l:
            if i == casteditem:
                return True
        else:
            return False
    return impl


@overload_method(types.ListType, 'count')
def impl_count(l, item):
    if not isinstance(l, types.ListType):
        return

    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        total = 0
        for i in l:
            if i == casteditem:
                total += 1
        return total

    return impl


@overload_method(types.ListType, 'extend')
def impl_extend(l, iterable):
    if not isinstance(l, types.ListType):
        return
    if not isinstance(iterable, types.IterableType):
        raise TypingError("extend argument must be iterable")

    def select_impl():
        if isinstance(iterable, types.ListType):
            def impl(l, iterable):
                # guard against l.extend(l)
                if l is iterable:
                    iterable = iterable.copy()
                for i in iterable:
                    l.append(i)

            return impl
        else:
            def impl(l, iterable):
                for i in iterable:
                    l.append(i)

            return impl

    if l.is_precise():
        # Handle the precise case.
        return select_impl()
    else:
        # Handle the imprecise case, try to 'guess' the underlying type of the
        # values in the iterable.
        if hasattr(iterable, "dtype"):  # tuples and arrays
            ty = iterable.dtype
        elif hasattr(iterable, "item_type"):  # lists
            ty = iterable.item_type
        else:
            raise TypingError("unable to extend list, iterable is missing "
                              "either *dtype* or *item_type*")
        l = l.refine(ty)
        # Create the signature that we wanted this impl to have
        sig = typing.signature(types.void, l, iterable)
        return sig, select_impl()


@overload_method(types.ListType, 'insert')
def impl_insert(l, index, item):
    if not isinstance(l, types.ListType):
        return

    if index in types.signed_domain:
        def impl(l, index, item):
            # If the index is larger than the size of the list or if the list is
            # empty, just append.
            if index >= len(l) or len(l) == 0:
                l.append(item)
            # Else, do the insert dance
            else:
                # convert negative indices
                if index < 0:
                    # if the index is still negative after conversion, use 0
                    index = max(len(l) + index, 0)
                # grow the list by one, make room for item to insert
                l.append(l[0])
                # reverse iterate over the list and shift all elements
                i = len(l) - 1
                while(i > index):
                    l[i] = l[i - 1]
                    i -= 1
                # finally, insert the item
                l[index] = item

        if l.is_precise():
            # Handle the precise case.
            return impl
        else:
            # Handle the imprecise case
            l = l.refine(item)
            # Re-bind the item type to match the arguments.
            itemty = l.item_type
            # Create the signature that we wanted this impl to have.
            sig = typing.signature(types.void, l, INDEXTY, itemty)
            return sig, impl
    else:
        raise TypingError("list insert indices must be signed integers")


@overload_method(types.ListType, 'remove')
def impl_remove(l, item):
    if not isinstance(l, types.ListType):
        return

    itemty = l.item_type

    def impl(l, item):
        casteditem = _cast(item, itemty)
        for i, n in enumerate(l):
            if casteditem == n:
                l.pop(i)
                return
        else:
            raise ValueError("list.remove(x): x not in list")

    return impl


@overload_method(types.ListType, 'clear')
def impl_clear(l):
    if not isinstance(l, types.ListType):
        return

    def impl(l):
        while len(l):
            l.pop()

    return impl


@overload_method(types.ListType, 'reverse')
def impl_reverse(l):
    if not isinstance(l, types.ListType):
        return

    def impl(l):
        front = 0
        back = len(l) - 1
        while front < back:
            l[front], l[back] = l[back], l[front]
            front += 1
            back -= 1

    return impl


@overload_method(types.ListType, 'copy')
def impl_copy(l):
    if isinstance(l, types.ListType):
        def impl(l):
            return l[:]

        return impl


@overload_method(types.ListType, 'index')
def impl_index(l, item, start=None, end=None):
    if not isinstance(l, types.ListType):
        return
    itemty = l.item_type

    def check_arg(arg, name):
        if not (arg is None
                or arg in types.signed_domain
                or isinstance(arg, (types.Omitted, types.NoneType))):
            raise TypingError("{} argument for index must be a signed integer"
                              .format(name))
    check_arg(start, "start")
    check_arg(end, "end")

    def impl(l, item, start=None, end=None):
        casteditem = _cast(item, itemty)
        for i in handle_slice(l, slice(start, end, 1)):
            if l[i] == casteditem:
                return i
        else:
            raise ValueError("item not in list")

    return impl


@register_jitable
def compare(this, other):
    """Oldschool (python 2.x) cmp.

       if this < other return -1
       if this = other return 0
       if this > other return 1
    """
    if len(this) != len(other):
        return -1 if len(this) < len(other) else 1
    for i in range(len(this)):
        this_item, other_item = this[i], other[i]
        if this_item != other_item:
            return -1 if this_item < other_item else 1
    else:
        return 0


def compare_helper(this, other, accepted):
    if not isinstance(this, types.ListType):
        return
    if not isinstance(other, types.ListType):
        raise TypingError("list can only be compared to list")

    def impl(this, other):
        return compare(this, other) in accepted
    return impl


@overload(operator.eq)
def impl_equal(this, other):
    return compare_helper(this, other, (0,))


@overload(operator.ne)
def impl_not_equal(this, other):
    return compare_helper(this, other, (-1, 1))


@overload(operator.lt)
def impl_less_than(this, other):
    return compare_helper(this, other, (-1, ))


@overload(operator.le)
def impl_less_than_or_equal(this, other):
    return compare_helper(this, other, (-1, 0))


@overload(operator.gt)
def impl_greater_than(this, other):
    return compare_helper(this, other, (1,))


@overload(operator.ge)
def impl_greater_than_or_equal(this, other):
    return compare_helper(this, other, (0, 1))


@lower_builtin('getiter', types.ListType)
def impl_list_getiter(context, builder, sig, args):
    """Implement iter(List).
    """
    [tl] = sig.args
    [l] = args
    iterablety = types.ListTypeIterableType(tl)
    it = context.make_helper(builder, iterablety.iterator_type)

    fnty = ir.FunctionType(
        ir.VoidType(),
        [ll_listiter_type, ll_list_type],
    )

    fn = builder.module.get_or_insert_function(fnty, name='numba_list_iter')

    proto = ctypes.CFUNCTYPE(ctypes.c_size_t)
    listiter_sizeof = proto(_helperlib.c_helpers['list_iter_sizeof'])
    state_type = ir.ArrayType(ir.IntType(8), listiter_sizeof())

    pstate = cgutils.alloca_once(builder, state_type, zfill=True)
    it.state = _as_bytes(builder, pstate)
    it.parent = l

    dp = _container_get_data(context, builder, iterablety.parent, args[0])
    builder.call(fn, [it.state, dp])
    return impl_ret_borrowed(
        context,
        builder,
        sig.return_type,
        it._getvalue(),
    )


@lower_builtin('iternext', types.ListTypeIteratorType)
@iternext_impl(RefType.BORROWED)
def impl_iterator_iternext(context, builder, sig, args, result):
    iter_type = sig.args[0]
    it = context.make_helper(builder, iter_type, args[0])

    iternext_fnty = ir.FunctionType(
        ll_status,
        [ll_listiter_type, ll_bytes.as_pointer()]
    )
    iternext = builder.module.get_or_insert_function(
        iternext_fnty,
        name='numba_list_iter_next',
    )
    item_raw_ptr = cgutils.alloca_once(builder, ll_bytes)

    status = builder.call(iternext, (it.state, item_raw_ptr))

    # check for list mutation
    mutated_status = status.type(int(ListStatus.LIST_ERR_MUTATED))
    is_mutated = builder.icmp_signed('==', status, mutated_status)
    with builder.if_then(is_mutated, likely=False):
        context.call_conv.return_user_exc(
            builder, RuntimeError, ("list was mutated during iteration",))

    # if the list wasn't mutated it is either fine or the iterator was
    # exhausted
    ok_status = status.type(int(ListStatus.LIST_OK))
    is_valid = builder.icmp_signed('==', status, ok_status)
    result.set_valid(is_valid)

    with builder.if_then(is_valid, likely=True):
        item_ty = iter_type.parent.item_type

        dm_item = context.data_model_manager[item_ty]

        item_ptr = builder.bitcast(
            builder.load(item_raw_ptr),
            dm_item.get_data_type().as_pointer(),
        )

        item = dm_item.load_from_data_pointer(builder, item_ptr)

        if isinstance(iter_type.iterable, ListTypeIterableType):
            result.yield_(item)
        else:
            # unreachable
            raise AssertionError('unknown type: {}'.format(iter_type.iterable))
