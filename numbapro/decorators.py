from numbapro import translate
from numba import double
import llvm.core as _lc

default_module = _lc.Module.new('default')
translates = []

def export(restype=double, argtypes=[double], backend='bytecode', **kws):
    def _export(func, name=None):
        # XXX: need to implement ast backend.
        t = translate.Translate(func, restype=restype,
                                argtypes=argtypes,
                                module=default_module,
                                name=name, **kws)
        t.translate()
        translates.append((t, name))
    return _export
