__all__ = [
    'vectorize',
    'guvectorize',
    'Vectorize',
    'BasicVectorize',
    'ParallelVectorize',
    'StreamVectorize',
    'GUFuncVectorize',
    'GUFuncASTVectorize',
    'CudaASTVectorize',
    'CudaGUFuncASTVectorize',
    'MiniVectorize',
    'ParallelMiniVectorize',
]
import logging
from numba.vectorize import *
from numba.vectorize import install_vectorizer, _prepare_sig

from .parallel import ParallelVectorize, ParallelASTVectorize
from .stream import StreamVectorize, StreamASTVectorize
from .gufunc import GUFuncVectorize, GUFuncASTVectorize
from numbapro.cudapipeline.error import CudaSupportError

try:
    from .cuda import  CudaASTVectorize, CudaGUFuncASTVectorize
except CudaSupportError, e:
    logging.warning("Cuda vectorizers not available, using fallbacks")
    CudaVectorize = BasicVectorize
    CUDAGUFuncVectorize = GUFuncVectorize
    CudaGUFuncASTVectorize = GUFuncASTVectorize
    CudaASTVectorize = BasicVectorize

from .minivectorize import MiniVectorize, ParallelMiniVectorize

install_vectorizer('bytecode', 'parallel', ParallelVectorize)
install_vectorizer('bytecode', 'stream', StreamVectorize)

install_vectorizer('ast', 'parallel', ParallelASTVectorize)
install_vectorizer('ast', 'stream', StreamASTVectorize)
install_vectorizer('ast', 'gpu', CudaASTVectorize)

install_vectorizer('mini', 'cpu', MiniVectorize)
install_vectorizer('mini', 'parallel', ParallelMiniVectorize)

_bytecode_guvectorizers = {
    'cpu': GUFuncVectorize,
}

_ast_guvectorizers = {
    'cpu': GUFuncASTVectorize,
    'gpu': CudaGUFuncASTVectorize,
}

_guvectorizers = {
    'bytecode': _bytecode_guvectorizers,
    'ast':      _ast_guvectorizers,
}

def GUVectorize(func, signature, backend='ast', target='cpu'):
    assert backend in _guvectorizers
    targets = _guvectorizers[backend]
    assert target in targets
    return targets[target](func, signature)

def guvectorize(fnsigs, gusig, backend='ast', target='cpu'):
    def _guvectorize(fn):
        vect = GUVectorize(fn, gusig, backend=backend, target=target)
        for sig in fnsigs:
            kws = _prepare_sig(sig)
            vect.add(**kws)
        ufunc = vect.build_ufunc()
        return ufunc

    return _guvectorize
