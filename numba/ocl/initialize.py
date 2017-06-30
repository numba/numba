from __future__ import absolute_import, print_function


def init_jit():
    from numba.ocl.dispatcher import OCLDispatcher
    return OCLDispatcher

def initialize_all():
    from numba.targets.registry import dispatcher_registry
    dispatcher_registry.ondemand['ocl'] = init_jit


def _initialize_ufunc():
    from numba.npyufunc import Vectorize

    def init_vectorize():
        from numba.ocl.vectorizers import OclVectorize

        return OclVectorize

    Vectorize.target_registry.ondemand['ocl'] = init_vectorize

def _initialize_gufunc():
    from numba.npyufunc import GUVectorize

    def init_guvectorize():
        from numba.ocl.vectorizers import OclGUFuncVectorize

        return OclGUFuncVectorize

    GUVectorize.target_registry.ondemand['ocl'] = init_guvectorize

