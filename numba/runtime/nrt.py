from __future__ import print_function, absolute_import, division

from . import _nrt_python as _nrt
from . import atomicops


class Runtime(object):
    def __init__(self):
        # Compile atomic operations
        compiled = atomicops.compile_atomic_ops()
        self._mcjit, self._ptr_inc, self._ptr_dec, self._ptr_cas = compiled
        # Install atomic ops to NRT
        _nrt.memsys_set_atomic_inc_dec(self._ptr_inc, self._ptr_dec)
        _nrt.memsys_set_atomic_cas(self._ptr_cas)

    @staticmethod
    def shutdown():
        _nrt.memsys_shutdown()

    def meminfo_new(self, data, pyobj):
        """
        Returns a MemInfo object that tracks memory at `data` owned by `pyobj`.
        MemInfo will acquire a reference on `pyobj`.
        The release of MemInfo will release a reference on `pyobj`.
        """
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
        if safe:
            mi = _nrt.meminfo_alloc_safe(size)
        else:
            mi = _nrt.meminfo_alloc(size)
        return MemInfo(mi)

    def process_defer_dtor(self):
        """Process all deferred dtors.
        """
        _nrt.memsys_process_defer_dtor()


MemInfo = _nrt._MemInfo
