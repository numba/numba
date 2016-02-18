"""
Support for native homogenous lists.
"""

from __future__ import print_function, absolute_import, division

import collections
import contextlib
import math

from llvmlite import ir
from numba import types, cgutils, typing
from numba.targets.imputils import (lower_builtin, lower_cast,
                                    iternext_impl, impl_ret_borrowed,
                                    impl_ret_new_ref, impl_ret_untracked)
from numba.utils import cached_property
from . import quicksort, slicing


def make_set_cls(set_type):
    """
    Return the Structure representation of the given *set_type*
    (an instance of types.Set).
    """
    return cgutils.create_struct_proxy(set_type)


def make_payload_cls(set_type):
    """
    Return the Structure representation of the given *set_type*'s payload
    (an instance of types.Set).
    """
    # Note the payload is stored durably in memory, so we consider it
    # data and not value.
    return cgutils.create_struct_proxy(types.SetPayload(set_type),
                                       kind='data')


def make_entry_cls(set_type):
    return cgutils.create_struct_proxy(types.SetEntry(set_type), kind='data')


def get_payload(context, builder, set_type, value):
    """
    Given a set value and type, get its payload structure (as a
    reference, so that mutations are seen by all).
    """
    payload_type = context.get_data_type(types.SetPayload(set_type))
    payload = context.nrt_meminfo_data(builder, value.meminfo)
    payload = builder.bitcast(payload, payload_type.as_pointer())
    return make_payload_cls(set_type)(context, builder, ref=payload)


def get_entry_size(context, set_type):
    """
    Return the entry size for the given set type.
    """
    llty = context.get_data_type(types.SetPayload(set_type))
    return context.get_abi_sizeof(llty)


# Note these values are special:
# - EMPTY is obtained by issuing memset(..., 0xFF)
# - (unsigned) EMPTY > (unsigned) DELETED > any other hash value
EMPTY = -1
DELETED = -2
FALLBACK = -43

MINSIZE = 4
#MINSIZE = 16


def get_hash_value(context, builder, typ, value):
    """
    Compute the hash of the given value.
    """
    sig = typing.signature(types.intp, typ)
    fn = context.get_function(hash, sig)
    h = fn(builder, (value,))
    # Fixup reserved values
    is_ok = is_hash_used(context, builder, h)
    fallback = ir.Constant(h.type, FALLBACK)
    return builder.select(is_ok, h, fallback)

def is_hash_empty(context, builder, h):
    """
    Whether the hash value denotes an empty entry.
    """
    empty = ir.Constant(h.type, EMPTY)
    return builder.icmp_unsigned('==', h, empty)

def is_hash_deleted(context, builder, h):
    """
    Whether the hash value denotes a deleted entry.
    """
    deleted = ir.Constant(h.type, DELETED)
    return builder.icmp_unsigned('==', h, deleted)

def is_hash_used(context, builder, h):
    """
    Whether the hash value denotes an active entry.
    """
    # Everything below DELETED is an used entry
    deleted = ir.Constant(h.type, DELETED)
    return builder.icmp_unsigned('<', h, deleted)


SetLoop = collections.namedtuple('SetLoop', ('entry', 'do_break'))


class _SetPayloadMixin(object):

    @property
    def mask(self):
        return self._payload.mask

    @mask.setter
    def mask(self, value):
        # CAUTION: mask must be a power of 2 minus 1
        self._payload.mask = value

    @property
    def used(self):
        return self._payload.used

    @used.setter
    def used(self, value):
        self._payload.used = value

    @property
    def fill(self):
        return self._payload.fill

    @fill.setter
    def fill(self, value):
        self._payload.fill = value

    @property
    def entries(self):
        return self._payload._get_ptr_by_name('entries')

    def get_entry(self, entries, idx):
        """
        Get entry number *idx* in the *entries* table.
        """
        ptr = cgutils.gep(self._builder, entries, idx)
        entry = make_entry_cls(self._ty)(self._context, self._builder, ref=ptr)
        return entry


class SetInstance(_SetPayloadMixin):

    def __init__(self, context, builder, set_type, set_val):
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._set = make_set_cls(set_type)(context, builder, set_val)
        self._entrysize = get_entry_size(context, set_type)
        self._entrymodel = context.data_model_manager[types.SetPayload(set_type)]
        self._datamodel = context.data_model_manager[set_type.dtype]

    @property
    def dtype(self):
        return self._ty.dtype

    # XXX set ._payload manually when resizing to avoid generating too much IR?

    @property
    def _payload(self):
        # This cannot be cached as it can be reallocated!
        return get_payload(self._context, self._builder, self._ty, self._set)

    @property
    def value(self):
        return self._set._getvalue()

    @property
    def meminfo(self):
        return self._set.meminfo

    def _lookup(self, entries, item, h):
        context = self._context
        builder = self._builder

        intp_t = h.type

        mask = self.mask
        dtype = self._ty.dtype
        eqfn = context.get_function('==',
                                    typing.signature(types.boolean, dtype, dtype))

        one = ir.Constant(intp_t, 1)
        five = ir.Constant(intp_t, 5)

        # The perturbation value for probing
        perturb = cgutils.alloca_once_value(builder, h)
        # The index of the entry being considered: start with (hash & mask)
        index = cgutils.alloca_once_value(builder,
                                          builder.and_(h, mask))

        bb_body = builder.append_basic_block("lookup.body")
        bb_found = builder.append_basic_block("lookup.found")
        bb_not_found = builder.append_basic_block("lookup.not_found")
        bb_end = builder.append_basic_block("lookup.end")

        builder.branch(bb_body)

        with builder.goto_block(bb_body):
            i = builder.load(index)
            entry = self.get_entry(entries, i)
            entry_hash = entry.hash

            with builder.if_then(builder.icmp_unsigned('==', h, entry_hash)):
                # Hashes are equal, compare values
                # (note this also ensures the entry is used)
                eq = eqfn(builder, (item, entry.value))
                with builder.if_then(eq):
                    builder.branch(bb_found)

            with builder.if_then(is_hash_empty(context, builder, entry_hash)):
                builder.branch(bb_not_found)

            # Perturb to go to next entry:
            #   perturb >>= 5
            #   i = (i * 5 + 1 + perturb) & mask
            p = builder.load(perturb)
            p = builder.lshr(p, five)
            i = builder.add(one, builder.mul(i, five))
            i = builder.and_(mask, builder.add(i, p))
            builder.store(i, index)
            builder.store(p, perturb)
            # Loop
            builder.branch(bb_body)

        with builder.goto_block(bb_not_found):
            builder.branch(bb_end)

        with builder.goto_block(bb_found):
            builder.branch(bb_end)

        builder.position_at_end(bb_end)

        found = builder.phi(ir.IntType(1), 'found')
        found.add_incoming(cgutils.true_bit, bb_found)
        found.add_incoming(cgutils.false_bit, bb_not_found)

        return found, builder.load(index)

    @contextlib.contextmanager
    def _iterate(self, entries, size):
        """
        Iterate over the *size* *entries*.
        A SetLoop is yielded.
        """
        context = self._context
        builder = self._builder

        intp_t = context.get_value_type(types.intp)
        one = ir.Constant(intp_t, 1)

        with cgutils.for_range(builder, size) as range_loop:
            entry = self.get_entry(entries, range_loop.index)
            is_used = is_hash_used(context, builder, entry.hash)
            with builder.if_then(is_used):
                loop = SetLoop(entry=entry, do_break=range_loop.do_break)
                yield loop

    def _add_entry(self, entries, item, h, do_resize=True):
        context = self._context
        builder = self._builder

        found, i = self._lookup(entries, item, h)
        not_found = builder.not_(found)

        with builder.if_then(not_found):
            # Not found => add it
            entry = self.get_entry(entries, i)
            old_hash = entry.hash
            entry.hash = h
            entry.value = item
            # used++
            used = self.used
            one = ir.Constant(used.type, 1)
            used = self.used = builder.add(used, one)
            # fill++ if entry wasn't a deleted one
            with builder.if_then(is_hash_empty(context, builder, old_hash),
                                 likely=True):
                self.fill = builder.add(self.fill, one)
            # Grow table if necessary
            if do_resize:
                self.upsize(used)

    def add(self, item):
        context = self._context
        builder = self._builder

        entries = self.entries
        h = get_hash_value(context, builder, self._ty.dtype, item)
        self._add_entry(entries, item, h)

    def contains(self, item):
        context = self._context
        builder = self._builder

        entries = self.entries
        h = get_hash_value(context, builder, self._ty.dtype, item)
        found, i = self._lookup(entries, item, h)
        return found

    @classmethod
    def allocate_ex(cls, context, builder, set_type, nitems=None):
        """
        Allocate a SetInstance with its storage.
        Return a (ok, instance) tuple where *ok* is a LLVM boolean and
        *instance* is a SetInstance object (the object's contents are
        only valid when *ok* is true).
        """
        intp_t = context.get_value_type(types.intp)

        # XXX for now, ignore nitems and choose a default value
        nentries = ir.Constant(intp_t, MINSIZE)

        self = cls(context, builder, set_type, None)
        ok = self._allocate_payload(nentries)
        return ok, self

    @classmethod
    def allocate(cls, context, builder, set_type, nitems=None):
        """
        Allocate a SetInstance with its storage.  Same as allocate_ex(),
        but return an initialized *instance*.  If allocation failed,
        control is transferred to the caller using the target's current
        call convention.
        """
        ok, self = cls.allocate_ex(context, builder, set_type, nitems)
        with builder.if_then(builder.not_(ok), likely=False):
            context.call_conv.return_user_exc(builder, MemoryError,
                                              ("cannot allocate set",))
        return self

    def upsize(self, nitems):
        """
        When adding to the set, ensure it is properly sized for the given
        number of used entries.
        """
        context = self._context
        builder = self._builder
        intp_t = nitems.type

        one = ir.Constant(intp_t, 1)
        two = ir.Constant(intp_t, 2)

        # Ensure number of entries >= 2 * used
        min_entries = builder.shl(nitems, one)
        size = builder.add(self.mask, one)
        need_resize = builder.icmp_unsigned('>=', min_entries, size)

        with builder.if_then(need_resize, likely=False):
            # Find out next suitable size
            new_size_p = cgutils.alloca_once_value(builder, size)

            bb_body = builder.append_basic_block("calcsize.body")
            bb_end = builder.append_basic_block("calcsize.end")

            builder.branch(bb_body)

            with builder.goto_block(bb_body):
                # Multiply by 4 (ensuring size remains a power of two)
                new_size = builder.load(new_size_p)
                new_size = builder.shl(new_size, two)
                builder.store(new_size, new_size_p)
                is_too_small = builder.icmp_unsigned('>=', min_entries, new_size)
                builder.cbranch(is_too_small, bb_body, bb_end)

            builder.position_at_end(bb_end)

            new_size = builder.load(new_size_p)

            #context.printf(builder, "upsize to %zd items: mask = %zd, min entries = %zd, new size = %zd\n",
                           #(nitems, self.mask, min_entries, new_size))

            # Allocate new entries
            old_entries = self.entries
            old_size = size
            old_ptr = context.nrt_meminfo_data(builder, self.meminfo)

            ok = self._allocate_payload(new_size, realloc=True)
            with builder.if_then(builder.not_(ok), likely=False):
                context.call_conv.return_user_exc(builder, MemoryError,
                                                  ("cannot grow set",))

            # Re-insert old entries
            entries = self.entries
            with self._iterate(old_entries, old_size) as loop:
                entry = loop.entry
                self._add_entry(entries, entry.value, entry.hash,
                                do_resize=False)

            self._free_payload(old_ptr)

    def _allocate_payload(self, nentries, realloc=False):
        """
        Allocate and initialize payload for the given number of entries.
        If *realloc* is True, the existing meminfo is reused.

        CAUTION: *nentries* must be a power of 2!
        """
        context = self._context
        builder = self._builder

        ok = cgutils.alloca_once_value(builder, cgutils.true_bit)

        intp_t = context.get_value_type(types.intp)
        zero = ir.Constant(intp_t, 0)
        one = ir.Constant(intp_t, 1)

        payload_type = context.get_data_type(types.ListPayload(self._ty))
        payload_size = context.get_abi_sizeof(payload_type)

        entry_size = get_entry_size(context, self._ty)

        # Total allocation size = <payload header size> + nentries * entry_size
        allocsize, ovf = cgutils.muladd_with_overflow(builder, nentries,
                                                      ir.Constant(intp_t, entry_size),
                                                      ir.Constant(intp_t, payload_size))
        with builder.if_then(ovf, likely=False):
            builder.store(cgutils.false_bit, ok)

        with builder.if_then(builder.load(ok), likely=True):
            if realloc:
                meminfo = self._set.meminfo
                ptr = context.nrt_meminfo_varsize_alloc(builder, meminfo,
                                                        size=allocsize)
                alloc_ok = cgutils.is_null(builder, ptr)
            else:
                meminfo = context.nrt_meminfo_new_varsize(builder, size=allocsize)
                alloc_ok = cgutils.is_null(builder, meminfo)

            with builder.if_else(cgutils.is_null(builder, meminfo),
                                 likely=False) as (if_error, if_ok):
                with if_error:
                    builder.store(cgutils.false_bit, ok)
                with if_ok:
                    if not realloc:
                        self._set.meminfo = meminfo
                    # Initialize entries to 0xff (EMPTY)
                    ptr = context.nrt_meminfo_data(builder, meminfo)
                    cgutils.memset(builder, ptr, allocsize, 0xFF)
                    payload = self._payload
                    payload.used = zero
                    payload.fill = zero
                    new_mask = builder.sub(nentries, one)
                    #context.printf(builder,
                                   #"allocated %zd bytes for set at %p: setting mask to %zd\n",
                                   #(allocsize, ptr, new_mask))
                    payload.mask = new_mask

        return builder.load(ok)

    def _free_payload(self, ptr):
        """
        Allocate an old payload at *ptr*.
        """
        self._context.nrt_meminfo_varsize_free(self._builder, self.meminfo, ptr)


#-------------------------------------------------------------------------------
# Constructors

@lower_builtin(set, types.IterableType)
def set_constructor(context, builder, sig, args):
    set_type = sig.return_type
    items_type, = sig.args

    inst = SetInstance.allocate(context, builder, set_type)
    # Populate set
    update_sig = typing.signature(types.none, set_type, items_type)
    update_args = (inst.value, args[0])
    set_update(context, builder, update_sig, update_args)

    return impl_ret_new_ref(context, builder, set_type, inst.value)



#-------------------------------------------------------------------------------
# Various operations

@lower_builtin(len, types.Set)
def set_len(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[0], args[0])
    return inst.used

@lower_builtin("in", types.Any, types.Set)
def in_set(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[1], args[1])
    return inst.contains(args[0])


#-------------------------------------------------------------------------------
# Methods

@lower_builtin("set.add", types.Set, types.Any)
def set_add(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[0], args[0])
    item = args[1]

    n = inst.used
    inst.add(item)

    return context.get_dummy_value()

#@lower_builtin("list.clear", types.List)
#def list_clear(context, builder, sig, args):
    #inst = SetInstance(context, builder, sig.args[0], args[0])
    #inst.resize(context.get_constant(types.intp, 0))

    #return context.get_dummy_value()

#@lower_builtin("list.copy", types.List)
#def list_copy(context, builder, sig, args):
    #def list_copy_impl(lst):
        #return list(lst)

    #return context.compile_internal(builder, list_copy_impl, sig, args)

#def _list_extend_list(context, builder, sig, args):
    #src = SetInstance(context, builder, sig.args[1], args[1])
    #dest = SetInstance(context, builder, sig.args[0], args[0])

    #src_size = src.size
    #dest_size = dest.size
    #nitems = builder.add(src_size, dest_size)
    #dest.resize(nitems)
    #dest.size = nitems

    #with cgutils.for_range(builder, src_size) as loop:
        #value = src.getitem(loop.index)
        #value = context.cast(builder, value, src.dtype, dest.dtype)
        #dest.setitem(builder.add(loop.index, dest_size), value)

    #return dest

@lower_builtin("set.update", types.Set, types.IterableType)
def set_update(context, builder, sig, args):
    def set_update(cont, iterable):
        # Speed hack to avoid NRT refcount operations inside the loop
        meth = cont.add
        for v in iterable:
            meth(v)

    return context.compile_internal(builder, set_update, sig, args)
