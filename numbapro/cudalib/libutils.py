import sys
from ctypes import *

class ctype_function(object):
    def __init__(self, restype=None, *argtypes):
        self.restype = restype
        self.argtypes = argtypes

class Lib(object):
    __singleton = None
    lib = None

    def __new__(cls, override_path=None):
        # Check if we already have opened the dll
        if cls.__singleton is None:
            # No

            # Determine dll extension type for the platform
            if sys.platform == 'win32':
                dlext = '.dll'
                dllopener = WinDLL
            elif sys.platform == 'darwin':
                dlext = '.dylib'
                dllopener = CDLL
            else:
                dlext = '.so'
                dllopener = CDLL
            # Open the DLL
            path = cls.lib + dlext if not override_path else override_path
            dll = dllopener(path)

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
                setattr(self, name, self._auto_checking_wrapper(fn))

    def _auto_checking_wrapper(self, fn):
        def wrapped(*args, **kws):
            status = fn(*args, **kws)
            self.check_error(status)
            return status
        return wrapped

    def check_error(self, status):
        if status != 0:
            raise self.ErrorType(status)

