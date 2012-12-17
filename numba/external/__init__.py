from external import ExternalLibrary

__all__ = []
all = {}

def _import_all():
    global __all__
    mods = ['pyapi',
            'libc']
    for k in mods:
        mod = __import__(__name__ + '.' + k, fromlist=['__all__'])
        __all__.extend(mod.__all__)
        for k in mod.__all__:
            all[k] = globals()[k] = getattr(mod, k)

_import_all()

def default_external_library(context):
    '''Build an external functino library
        
    context --- numba context
    '''
    extlib = ExternalLibrary(context)
    
    for fncls in all.itervalues():
        extlib.add(fncls())
    return extlib