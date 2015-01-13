from __future__ import absolute_import
from numba.cuda.cudadrv.libs import open_cudalib

class ctype_function(object):
    def __init__(self, restype=None, *argtypes):
        self.restype = restype
        self.argtypes = argtypes

class Lib(object):
    __singleton = None
    lib = None

    def __new__(cls):
        # Check if we already have opened the dll
        if cls.__singleton is None:
            try:
                dll = open_cudalib(cls.lib)
            except OSError as e:
                raise Exception("Cannot open library for %s:\n%s" % (cls.lib,
                                                                     e))
            # Create new instance
            inst = object.__new__(cls)
            cls.__singleton = inst
            inst.dll = dll
            inst._initialize()
        else:
            inst = cls.__singleton
        return inst

    def _initialize(self):
        # Populate the instance with the functions
        for name, obj in vars(type(self)).items():
            if isinstance(obj, ctype_function):
                fn = getattr(self.dll, name)
                fn.restype = obj.restype
                fn.argtypes = obj.argtypes
                setattr(self, name, self._auto_checking_wrapper(fn, name=name))

    def _auto_checking_wrapper(self, fn, name):
        def wrapped(*args, **kws):
            nargs = len(args) + len(kws)
            expected = len(fn.argtypes)
            if nargs != expected:
                msg = "expecting {expected} arguments but got {nargs}: {fname}"
                raise TypeError(msg.format(expected=expected, nargs=nargs,
                                           fname=name))
            status = fn(*args, **kws)
            self.check_error(status)
            return status
        return wrapped

    def check_error(self, status):
        if status != 0:
            raise self.ErrorType(status)

