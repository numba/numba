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


def get_payload_struct(context, builder, set_type, ptr):
    """
    Given a set value and type, get its payload structure (as a
    reference, so that mutations are seen by all).
    """
    payload_type = context.get_data_type(types.SetPayload(set_type))
    payload = builder.bitcast(ptr, payload_type.as_pointer())
    return make_payload_cls(set_type)(context, builder, ref=payload)


def get_entry_size(context, set_type):
    """
    Return the entry size for the given set type.
    """
    llty = context.get_data_type(types.SetEntry(set_type))
    return context.get_abi_sizeof(llty)


# Note these values are special:
# - EMPTY is obtained by issuing memset(..., 0xFF)
# - (unsigned) EMPTY > (unsigned) DELETED > any other hash value
EMPTY = -1
DELETED = -2
FALLBACK = -43

MINSIZE = 4
#MINSIZE = 16

DEBUG_ALLOCS = False


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


class _SetPayload(object):

    def __init__(self, context, builder, set_type, ptr):
        payload = get_payload_struct(context, builder, set_type, ptr)
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._payload = payload
        self._entries = payload._get_ptr_by_name('entries')
        self._ptr = ptr

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
        """
        A pointer to the start of the entries array.
        """
        return self._entries

    @property
    def ptr(self):
        """
        A pointer to the start of the NRT-allocated area.
        """
        return self._ptr

    def get_entry(self, idx):
        """
        Get entry number *idx*.
        """
        entry_ptr = cgutils.gep(self._builder, self._entries, idx)
        entry = make_entry_cls(self._ty)(self._context, self._builder,
                                         ref=entry_ptr)
        return entry

    def _lookup(self, item, h):
        """
        Lookup the *item* with the given hash values in the entries.

        Return a (found, entry index) tuple:
        - If found is true, <entry index> points to the entry containing
          the item.
        - If found is false, <entry index> points to the empty entry that
          the item can be written to.
        """
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
            entry = self.get_entry(i)
            entry_hash = entry.hash

            with builder.if_then(builder.icmp_unsigned('==', h, entry_hash)):
                # Hashes are equal, compare values
                # (note this also ensures the entry is used)
                eq = eqfn(builder, (item, entry.key))
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
    def _iterate(self):
        """
        Iterate over the payload's entries.  Yield a SetLoop.
        """
        context = self._context
        builder = self._builder

        intp_t = context.get_value_type(types.intp)
        one = ir.Constant(intp_t, 1)
        size = builder.add(self.mask, one)

        with cgutils.for_range(builder, size) as range_loop:
            entry = self.get_entry(range_loop.index)
            is_used = is_hash_used(context, builder, entry.hash)
            with builder.if_then(is_used):
                loop = SetLoop(entry=entry, do_break=range_loop.do_break)
                yield loop


class SetInstance(object):

    def __init__(self, context, builder, set_type, set_val):
        self._context = context
        self._builder = builder
        self._ty = set_type
        self._entrysize = get_entry_size(context, set_type)
        self._entrymodel = context.data_model_manager[types.SetPayload(set_type)]
        self._datamodel = context.data_model_manager[set_type.dtype]
        self._set = make_set_cls(set_type)(context, builder, set_val)

    @property
    def dtype(self):
        return self._ty.dtype

    @property
    def payload(self):
        """
        The _SetPayload for this set.
        """
        # This cannot be cached as the pointer can move around!
        context = self._context
        builder = self._builder

        ptr = self._context.nrt_meminfo_data(builder, self.meminfo)
        return _SetPayload(context, builder, self._ty, ptr)

    @property
    def value(self):
        return self._set._getvalue()

    @property
    def meminfo(self):
        return self._set.meminfo

    def _add_entry(self, payload, item, h, do_resize=True):
        context = self._context
        builder = self._builder

        found, i = payload._lookup(item, h)
        not_found = builder.not_(found)

        with builder.if_then(not_found):
            # Not found => add it
            entry = payload.get_entry(i)
            old_hash = entry.hash
            entry.hash = h
            entry.key = item
            # used++
            used = payload.used
            one = ir.Constant(used.type, 1)
            used = payload.used = builder.add(used, one)
            # fill++ if entry wasn't a deleted one
            with builder.if_then(is_hash_empty(context, builder, old_hash),
                                 likely=True):
                payload.fill = builder.add(payload.fill, one)
            # Grow table if necessary
            if do_resize:
                self.upsize(used)

    def add(self, item):
        context = self._context
        builder = self._builder

        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        self._add_entry(payload, item, h)

    def contains(self, item):
        context = self._context
        builder = self._builder

        payload = self.payload
        h = get_hash_value(context, builder, self._ty.dtype, item)
        found, i = payload._lookup(item, h)
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

        payload = self.payload

        # Ensure number of entries >= 2 * used
        min_entries = builder.shl(nitems, one)
        size = builder.add(payload.mask, one)
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

            if DEBUG_ALLOCS:
                context.printf(builder,
                               "upsize to %zd items: current size = %zd, "
                               "min entries = %zd, new size = %zd\n",
                               (nitems, size, min_entries, new_size))

            # Allocate new entries
            old_payload = payload

            ok = self._allocate_payload(new_size, realloc=True)
            with builder.if_then(builder.not_(ok), likely=False):
                context.call_conv.return_user_exc(builder, MemoryError,
                                                  ("cannot grow set",))

            # Re-insert old entries
            payload = self.payload
            with old_payload._iterate() as loop:
                entry = loop.entry
                self._add_entry(payload, entry.key, entry.hash,
                                do_resize=False)

            self._free_payload(old_payload.ptr)

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

        payload_type = context.get_data_type(types.SetPayload(self._ty))
        payload_size = context.get_abi_sizeof(payload_type)
        entry_size = get_entry_size(context, self._ty)
        # Account for the fact that the payload struct already contains an entry
        payload_size -= entry_size

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
                    payload = self.payload
                    # Initialize entries to 0xff (EMPTY)
                    cgutils.memset(builder, payload.ptr, allocsize, 0xFF)
                    payload.used = zero
                    payload.fill = zero
                    new_mask = builder.sub(nentries, one)
                    payload.mask = new_mask
                    if DEBUG_ALLOCS:
                        context.printf(builder,
                                       "allocated %zd bytes for set at %p: mask = %zd\n",
                                       (allocsize, payload.ptr, new_mask))

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

    # TODO: if the argument has a len(), preallocate the set so as to
    # avoid resizes

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
    return inst.payload.used

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
