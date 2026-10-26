"""
Concurrent (sharded) typed dictionary for parallel writes.

This module implements a thread-safe dictionary for use inside
``parallel=True`` jit functions, e.g. concurrent writes into a shared
dictionary from a ``prange`` loop (see issues #10967 and #8767).

The dictionary is *sharded*: it holds ``n_shards`` internal typed
dictionaries. Each key is mapped to exactly one shard via
``hash(key) % n_shards``. Every shard is protected by its own spinlock
(an atomic exchange-based lock on a 64-bit lock word), so concurrent
writers only contend when hashing to the same shard. With, for
example, 256 shards and 16 threads, the probability of contention is
low and writes scale well.

The number of shards is tunable through the ``n_shards`` argument of
:func:`ConcurrentDict.empty` / :func:`concurrent_dict`. A shard count
of 1 degenerates to a globally locked dictionary; larger counts reduce
contention at the cost of a slightly more expensive ``len()``.

Notes on semantics
------------------
* Writes and reads of a single key are serialized by the lock of the
  shard the key maps to; concurrent writes to the *same* key are
  last-writer-wins.
* ``len()`` acquires every shard lock in turn and returns the exact
  number of stored entries (for a concurrently modified dictionary it
  is a snapshot that may be outdated by the time it is returned).
* ``popitem()`` removes an arbitrary entry (whichever non-empty shard
  is encountered first, locking one shard at a time).
* ``copy()`` takes each shard lock in turn and returns a snapshot as a
  new ``ConcurrentDict``; the shard count of the copy can be changed
  through its optional argument.
* ``update()`` accepts a ``numba.typed.Dict``; the source dictionary is
  read-only for this operation and must not be concurrently modified.
* Key types are those supported by ``numba.typed.Dict`` for which the
  ``hash()`` builtin is supported (e.g. integers, floats, booleans,
  unicode strings).

Unsupported operations
----------------------
* The dictionary cannot be iterated, and consequently ``keys()``,
  ``values()``, ``items()``, equality comparison (``==``) and
  ``pop(key, default)`` (two-argument form) are not supported:
  iteration of a concurrently mutating dictionary is unsafe. Use
  ``ConcurrentDict`` where access happens by key, e.g. results
  collected by worker id.
"""
from threading import RLock as _pythreading_rlock

import numpy as np

from llvmlite import ir

from numba.core import cgutils, types
from numba.core.errors import NumbaTypeError
from numba.core.extending import intrinsic, overload
from numba.experimental.jitclass import jitclass
from numba.typed.dictobject import new_dict


# ---------------------------------------------------------------------------
# Spinlock primitives (CPU mutex)
#
# The lock is a single uint64 word. 0 means unlocked, 1 means locked.
# Acquisition is an atomic exchange of 1 with acquire ordering until the
# previous value was 0; release is an atomic exchange of 0 with release
# ordering.
# ---------------------------------------------------------------------------


def _check_lock_array(locks):
    if not isinstance(locks, types.Array) or locks.dtype != types.uint64:
        raise NumbaTypeError(
            "spinlock word array must be a uint64 array, got "
            f"{locks}"
        )
    if locks.ndim != 1:
        raise NumbaTypeError(
            "spinlock word array must be 1-D, got "
            f"{locks.ndim}-D"
        )


@intrinsic
def spin_lock(typingctx, locks, idx):
    """Acquire the spinlock at ``locks[idx]``; busy-waits until the lock
    is acquired.
    """
    _check_lock_array(locks)

    sig = types.none(locks, idx)

    def codegen(context, builder, sig, args):
        locks_arr, idx_val = args
        ary = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=locks_arr,
        )
        ll_word_ptr = context.get_value_type(types.uint64).as_pointer()
        word_ptr = builder.bitcast(ary.data, ll_word_ptr)
        word_ptr = builder.gep(word_ptr, [idx_val])

        one = ir.Constant(ir.IntType(64), 1)
        zero = ir.Constant(ir.IntType(64), 0)

        spin_bb = builder.append_basic_block("spin_lock_spin")
        acquired_bb = builder.append_basic_block("spin_lock_acquired")
        builder.branch(spin_bb)
        with builder.goto_block(spin_bb):
            old = builder.atomic_rmw("xchg", word_ptr, one,
                                     ordering="acquire")
            acquired = builder.icmp_unsigned("==", old, zero)
            builder.cbranch(acquired, acquired_bb, spin_bb)
        builder.position_at_end(acquired_bb)

    return sig, codegen


@intrinsic
def spin_unlock(typingctx, locks, idx):
    """Release the spinlock at ``locks[idx]``.
    """
    _check_lock_array(locks)

    sig = types.none(locks, idx)

    def codegen(context, builder, sig, args):
        locks_arr, idx_val = args
        ary = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=locks_arr,
        )
        ll_word_ptr = context.get_value_type(types.uint64).as_pointer()
        word_ptr = builder.bitcast(ary.data, ll_word_ptr)
        word_ptr = builder.gep(word_ptr, [idx_val])

        zero = ir.Constant(ir.IntType(64), 0)
        builder.atomic_rmw("xchg", word_ptr, zero, ordering="release")

    return sig, codegen


# ---------------------------------------------------------------------------
# Sharded dictionary
# ---------------------------------------------------------------------------

# Compile-time (typing-time) guard against creating an unreasonable
# number of jitclass specializations by accident; there is no inherent
# limit on the number of shards at runtime.
_MAX_CACHED_SPECIALIZATIONS = 256

_specialization_cache = {}
_specialization_cache_guard = _pythreading_rlock()


def _get_jit_class(key_type, value_type):
    """Return (and cache) the jitclass implementing the sharded
    dictionary for *key_type* / *value_type*.
    """
    cache_key = (str(key_type), str(value_type))
    with _specialization_cache_guard:
        impl = _specialization_cache.get(cache_key)
        if impl is None:
            if len(_specialization_cache) >= _MAX_CACHED_SPECIALIZATIONS:
                raise RuntimeError(
                    "too many distinct ConcurrentDict specializations; "
                    "the shard dictionary is parameterized by key/value "
                    "type at compile time"
                )
            shard_type = types.DictType(key_type, value_type)
            # The shards are stored in a typed dictionary mapping the
            # (integer) shard index to the shard dictionary itself. The
            # outer dictionary is fully populated at construction time
            # and only read afterwards, which is safe from multiple
            # threads without locking.
            outer_type = types.DictType(types.int64, shard_type)
            spec = [
                ('locks', types.uint64[::1]),
                ('shards', outer_type),
                ('n_shards', types.int64),
            ]

            @jitclass(spec)
            class ConcurrentDictImpl:

                def __init__(self, n_shards):
                    if n_shards < 1:
                        raise ValueError(
                            "n_shards must be at least 1"
                        )
                    self.n_shards = n_shards
                    self.locks = np.zeros(n_shards, dtype=np.uint64)
                    self.shards = new_dict(types.int64, shard_type)
                    for i in range(n_shards):
                        self.shards[i] = new_dict(key_type, value_type)

                def _shard_index(self, key):
                    return hash(key) % self.n_shards

                def __setitem__(self, key, value):
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    self.shards[idx][key] = value
                    spin_unlock(self.locks, idx)

                def __getitem__(self, key):
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    if key in self.shards[idx]:
                        val = self.shards[idx][key]
                        spin_unlock(self.locks, idx)
                        return val
                    else:
                        # Unlock before raising: an exception escaping the
                        # method would skip the unlock and permanently
                        # block this shard.
                        spin_unlock(self.locks, idx)
                        raise KeyError(key)

                def __delitem__(self, key):
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    if key in self.shards[idx]:
                        del self.shards[idx][key]
                        spin_unlock(self.locks, idx)
                    else:
                        # Unlock before raising (see __getitem__).
                        spin_unlock(self.locks, idx)
                        raise KeyError(key)

                def get(self, key, default):
                    """Return the value for *key*, or *default* if the
                    key is not present.
                    """
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    if key in self.shards[idx]:
                        val = self.shards[idx][key]
                    else:
                        val = default
                    spin_unlock(self.locks, idx)
                    return val

                def setdefault(self, key, default):
                    """If *key* is present, return its value. Otherwise
                    insert *default* under *key* and return it.
                    """
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    if key in self.shards[idx]:
                        val = self.shards[idx][key]
                    else:
                        self.shards[idx][key] = default
                        val = default
                    spin_unlock(self.locks, idx)
                    return val

                def update(self, other):
                    """Update the dictionary with the key/value pairs of
                    the typed dictionary *other*, overwriting existing
                    keys. *other* is read-only for this operation and
                    must not be concurrently modified.
                    """
                    for key, value in other.items():
                        idx = self._shard_index(key)
                        spin_lock(self.locks, idx)
                        self.shards[idx][key] = value
                        spin_unlock(self.locks, idx)

                def popitem(self):
                    """Remove and return an arbitrary ``(key, value)``
                    pair; raises KeyError if the dictionary is empty.
                    """
                    for i in range(self.n_shards):
                        spin_lock(self.locks, i)
                        if len(self.shards[i]) > 0:
                            keyval = self.shards[i].popitem()
                            spin_unlock(self.locks, i)
                            return keyval
                        spin_unlock(self.locks, i)
                    raise KeyError("popitem from an empty ConcurrentDict")

                def clear(self):
                    """Remove all entries.
                    """
                    for i in range(self.n_shards):
                        spin_lock(self.locks, i)
                        self.shards[i].clear()
                        spin_unlock(self.locks, i)

                def copy(self, n_shards=None):
                    """Return a new ConcurrentDict with the same key and
                    value types containing the same entries. The copy
                    takes each shard lock in turn, so it is safe against
                    concurrent writers (producing a snapshot). The
                    *n_shards* argument optionally selects a different
                    shard count for the copy; by default the shard count
                    is inherited.
                    """
                    if n_shards is None:
                        n_shards = self.n_shards
                    new = ConcurrentDictImpl(n_shards)
                    for i in range(self.n_shards):
                        spin_lock(self.locks, i)
                        for key, value in self.shards[i].items():
                            idx = new._shard_index(key)
                            new.shards[idx][key] = value
                        spin_unlock(self.locks, i)
                    return new

                def __contains__(self, key):
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    contains = key in self.shards[idx]
                    spin_unlock(self.locks, idx)
                    return contains

                def pop(self, key):
                    """Remove *key* and return its value; raises
                    KeyError if the key is not present.
                    """
                    idx = self._shard_index(key)
                    spin_lock(self.locks, idx)
                    if key in self.shards[idx]:
                        val = self.shards[idx].pop(key)
                        spin_unlock(self.locks, idx)
                        return val
                    else:
                        # Unlock before raising (see __getitem__).
                        spin_unlock(self.locks, idx)
                        raise KeyError(key)

                def __len__(self):
                    total = 0
                    for i in range(self.n_shards):
                        spin_lock(self.locks, i)
                        total += len(self.shards[i])
                        spin_unlock(self.locks, i)
                    return total

            _specialization_cache[cache_key] = ConcurrentDictImpl
            impl = ConcurrentDictImpl
    return impl


class ConcurrentDict:
    """Concurrent (sharded) thread-safe dictionary.

    Create with :meth:`ConcurrentDict.empty`. Inside jit functions use
    :func:`concurrent_dict`.
    """

    @classmethod
    def empty(cls, key_type, value_type, n_shards=256):
        """Create a concurrent dictionary with *key_type* keys and
        *value_type* values, spread over *n_shards* internally locked
        shards.
        """
        impl = _get_jit_class(key_type, value_type)
        return impl(n_shards)


def concurrent_dict(key_type, value_type, n_shards=256):
    """Create a concurrent dictionary with *key_type* keys and
    *value_type* values, spread over *n_shards* internally locked
    shards. Callable from Python and from jit functions.
    """
    return ConcurrentDict.empty(key_type, value_type, n_shards=n_shards)


@overload(concurrent_dict)
def _ol_concurrent_dict(key_type, value_type, n_shards=256):
    """Nopython implementation of concurrent_dict(). *key_type* and
    *value_type* arrive as TypeRef instances.
    """
    kt = key_type.instance_type
    vt = value_type.instance_type
    impl_class = _get_jit_class(kt, vt)

    def impl(key_type, value_type, n_shards=256):
        return impl_class(n_shards)

    return impl
