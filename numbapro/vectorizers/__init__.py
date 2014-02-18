__all__ = [
    'vectorize',
    'guvectorize',
    'Vectorize',
    'BasicVectorize',
    'ParallelVectorize',
    'StreamVectorize',
    'GUFuncVectorize',
    'GUFuncASTVectorize',
    'CudaVectorize',
    'CudaGUFuncVectorize',
]

from numba.npyufunc import Vectorize, GUVectorize, vectorize, guvectorize
#
# from .parallel import ParallelVectorize, ParallelASTVectorize
# from .stream import StreamVectorize, StreamASTVectorize
#
# GUFuncVectorize = GUVectorize
# GUFuncASTVectorize = GUVectorize
#
# from numbapro.cudavec.vectorizers import  CudaVectorize, CudaGUFuncVectorize
#
# install_vectorizer('ast', 'parallel', ParallelASTVectorize)
# install_vectorizer('ast', 'stream', StreamASTVectorize)
# install_vectorizer('ast', 'gpu', CudaVectorize)
#
# _ast_guvectorizers = {
#     'cpu': GUFuncVectorize,
#     'gpu': CudaGUFuncVectorize,
# }
#
# _guvectorizers = {
#     'ast':      _ast_guvectorizers,
# }
#
# def GUVectorize(func, signature, backend='ast', target='cpu'):
#     assert backend in _guvectorizers, "unsupported backend"
#     targets = _guvectorizers[backend]
#     assert target in targets, "unsupported target"
#     return targets[target](func, signature)
#
# def guvectorize(fnsigs, gusig, backend='ast', target='cpu'):
#     '''guvectorize(fnsigs, gusig[, target="cpu"])
#
#     Similar to ``numbapro.vectorize`` but creates a generalized-ufunc
#     (gufunc).  Takes an additional argument ``gusig`` to specify the gufunc
#     signature.
#
#     :param fnsigs: list of function type signature as function type object or
#                    as string.
#
#     :param gusig:  gufunc signature string.
#     :param target: "cpu" or "gpu"
#
#     Please see `NumPy docs <http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html#details-of-signature>`_
#     for details of gufunc signature.
#
#     Refer to http://docs.continuum.io/numbapro/quickstart.html for usage
#     '''
#     def _guvectorize(fn):
#         vect = GUVectorize(fn, gusig, backend=backend, target=target)
#         for sig in fnsigs:
#             kws = _prepare_sig(sig)
#             vect.add(**kws)
#         ufunc = vect.build_ufunc()
#         return ufunc
#
#     return _guvectorize
