from __future__ import print_function, absolute_import, division

from collections import namedtuple
from weakref import finalize as _finalize

from . import nrtdynmod
from llvmlite import binding as ll

from numba.compiler_lock import global_compiler_lock
from numba.typing.typeof import typeof_impl
from numba import types
from . import _nrt_python as _nrt

_nrt_mstats = namedtuple("nrt_mstats", ["alloc", "free", "mi_alloc", "mi_free"])


class _Runtime(object):
    def __init__(self):
        self._init = False

    @global_compiler_lock
    def initialize(self, ctx):
        """Initializes the NRT

        Must be called before any actual call to the NRT API.
        Safe to be called multiple times.
        """
        if self._init:
            # Already initialized
            return

        # Register globals into the system
        for py_name in _nrt.c_helpers:
            c_name = "NRT_" + py_name
            c_address = _nrt.c_helpers[py_name]
            ll.add_symbol(c_name, c_address)

        # Compile atomic operations
        self._library = nrtdynmod.compile_nrt_functions(ctx)

        self._ptr_inc = self._library.get_pointer_to_function("nrt_atomic_add")
        self._ptr_dec = self._library.get_pointer_to_function("nrt_atomic_sub")
        self._ptr_cas = self._library.get_pointer_to_function("nrt_atomic_cas")

        # Install atomic ops to NRT
        _nrt.memsys_set_atomic_inc_dec(self._ptr_inc, self._ptr_dec)
        _nrt.memsys_set_atomic_cas(self._ptr_cas)

        self._init = True

    def _init_guard(self):
        if not self._init:
            msg = "Runtime must be initialized before use."
            raise RuntimeError(msg)

    @staticmethod
    def shutdown():
        """
        Shutdown the NRT
        Safe to be called without calling Runtime.initialize first
        """
        _nrt.memsys_shutdown()

    @property
    def library(self):
        """
        Return the Library object containing the various NRT functions.
        """
        self._init_guard()
        return self._library

    def meminfo_new(self, data, pyobj):
        """
        Returns a MemInfo object that tracks memory at `data` owned by `pyobj`.
        MemInfo will acquire a reference on `pyobj`.
        The release of MemInfo will release a reference on `pyobj`.
        """
        self._init_guard()
        mi = _nrt.meminfo_new(data, pyobj)
        return MemInfo(mi)

    def meminfo_alloc(self, size, safe=False):
        """
        Allocate a new memory of `size` bytes and returns a MemInfo object
        that tracks the allocation.  When there is no more reference to the
        MemInfo object, the underlying memory will be deallocated.

        If `safe` flag is True, the memory is allocated using the `safe` scheme.
        This is used for debugging and testing purposes.
        See `NRT_MemInfo_alloc_safe()` in "nrt.h" for details.
        """
        self._init_guard()
        if safe:
            mi = _nrt.meminfo_alloc_safe(size)
        else:
            mi = _nrt.meminfo_alloc(size)
        return MemInfo(mi)

    def get_allocation_stats(self):
        """
        Returns a namedtuple of (alloc, free, mi_alloc, mi_free) for count of
        each memory operations.
        """
        # No init guard needed to access stats members
        return _nrt_mstats(alloc=_nrt.memsys_get_stats_alloc(),
                           free=_nrt.memsys_get_stats_free(),
                           mi_alloc=_nrt.memsys_get_stats_mi_alloc(),
                           mi_free=_nrt.memsys_get_stats_mi_free())


# Alias to _nrt_python._MemInfo
MemInfo = _nrt._MemInfo


@typeof_impl.register(MemInfo)
def typeof_meminfo(val, c):
    return types.MemInfoPointer(types.voidptr)


# Create runtime
_nrt.memsys_use_cpython_allocator()
rtsys = _Runtime()

# Install finalizer
_finalize(rtsys, _Runtime.shutdown)

# Avoid future use of the class
del _Runtime
