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

    def meminfo_new(self, data, pyobj):
        mi = _nrt.meminfo_new(data, pyobj)
        return MemInfo(mi)

    def meminfo_alloc(self, size, safe=False):
        if safe:
            mi = _nrt.meminfo_alloc_safe(size)
        else:
            mi = _nrt.meminfo_alloc(size)
        return MemInfo(mi)

    def process_defer_dtor(self):
        _nrt.memsys_process_defer_dtor()


class MemInfo(_nrt._MemInfo):
    """
    A wrapper of MemInfo object in NRT.
    Defines the buffer protocol.
    """
    __slots__ = ()

    @property
    def defer(self):
        return self.get_defer()

    @defer.setter
    def defer(self, enable):
        self.set_defer(enable)

    @property
    def data(self):
        return self.get_data()

