from __future__ import print_function, absolute_import, division

from . import _nrt_python as _nrt
from . import atomicops


class Runtime(object):
    def __init__(self):
        # Compile atomic operations
        compiled = atomicops.compile_atomic_ops()
        self._mcjit, self._ptr_inc, self._ptr_dec = compiled
        # Install atomic ops to NRT
        _nrt.memsys_set_atomic_inc_dec(self._ptr_inc, self._ptr_dec)

    def meminfo_new(self, data, pyobj):
        mi = _nrt.meminfo_new(data, pyobj)
        return MemInfo(mi)

    def process_defer_dtor(self):
        _nrt.memsys_process_defer_dtor()


class MemInfo(object):
    __slots__ = ['_pointer', '_default_defer']

    def __init__(self, ptr):
        self._pointer = ptr
        self._default_defer = False
        self.acquire()  # acquire

    def __del__(self):
        self.release()  # release

    @property
    def defer(self):
        return self._default_defer

    @defer.setter
    def defer(self, enable):
        self._default_defer = enable

    def acquire(self):
        _nrt.meminfo_acquire(self._pointer)

    def release(self, defer=None):
        _nrt.meminfo_release(self._pointer,
                             self._default_defer if defer is None else defer)

