import llvm.core
from intrinsic import IntrinsicLibrary

__all__ = []
all = {}

def _import_all():
    global __all__
    mods = ['math_intrinsic',
            'string_intrinsic']
    for k in mods:
        mod = __import__(__name__ + '.' + k, fromlist=['__all__'])
        __all__.extend(mod.__all__)
        for k in mod.__all__:
            all[k] = globals()[k] = getattr(mod, k)

_import_all()


def default_intrinsic_library(context):
    '''Build an intrinsic library with a default set of external functions.

    context --- numba context

    TODO: It is possible to cache the default intrinsic library as a bitcode
    file on disk so that we don't build it every time.
    '''    
    intrlib = IntrinsicLibrary(context)
    # install intrinsics
    for fncls in all.itervalues():
        intrlib.add(fncls)
    return intrlib

    