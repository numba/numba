#### Additional initialization code ######


def _initialize_ufunc():
    from numba.npyufunc import Vectorize

    def init_vectorize():
        from numba.hsa.vectorizers import HsaVectorize

        return HsaVectorize

    Vectorize.target_registry.ondemand['hsa'] = init_vectorize


def _initialize_gufunc():
    from numba.npyufunc import GUVectorize

    def init_guvectorize():
        from numba.hsa.vectorizers import HsaGUFuncVectorize

        return HsaGUFuncVectorize

    GUVectorize.target_registry.ondemand['hsa'] = init_guvectorize


_initialize_ufunc()
_initialize_gufunc()
