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


