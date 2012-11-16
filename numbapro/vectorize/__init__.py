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
from .basic import BasicVectorize, BasicASTVectorize
from .parallel import ParallelVectorize, ParallelASTVectorize
from .stream import StreamVectorize, StreamASTVectorize
from .gufunc import GUFuncVectorize, GUFuncASTVectorize
from numbapro._cuda.error import CudaSupportError
from numba.decorators import _process_sig

try:
    from .cuda import  CudaASTVectorize
    from .gufunc import CudaGUFuncASTVectorize
except CudaSupportError, e:
    logging.warning("Cuda vectorizers not available, using fallbacks")
    CudaVectorize = BasicVectorize
    CUDAGUFuncVectorize = GUFuncVectorize
    CudaGUFuncASTVectorize = GUFuncASTVectorize

from .minivectorize import MiniVectorize, ParallelMiniVectorize

vectorizers = {
    'cpu': BasicVectorize,
    'parallel': ParallelVectorize,
    'stream': StreamVectorize,
}

ast_vectorizers = {
    'cpu': BasicASTVectorize,
    'parallel': ParallelASTVectorize,
    'stream': StreamASTVectorize,
    'gpu': CudaASTVectorize,
}

mini_vectorizers = {
    'cpu': MiniVectorize,
    'parallel': ParallelMiniVectorize,
}

backends = {
    'bytecode': vectorizers,
    'ast': ast_vectorizers,
    'mini': mini_vectorizers,
}

def Vectorize(func, backend='ast', target='cpu'):
    """
    Instantiate a vectorizer given the backend and target.

    func: the function to vectorize
    backend: 'bytecode', 'ast' or 'mini'.
             Default: 'bytecode'
    target: 'basic', 'parallel', 'stream' or 'gpu'
            Default: 'basic'
    """
    assert backend in backends, tuple(backends)
    targets = backends[backend]
    assert target in targets, tuple(targets)

    if target in targets:
        return targets[target](func)
    else:
        # Use the default bytecode backend
        return vectorizers[target](func)

guvectorizers = {
    'cpu': GUFuncVectorize,
}

ast_guvectorizers = {
    'cpu': GUFuncASTVectorize,
    'gpu': CudaGUFuncASTVectorize,
}

guvectorizers_backends = {
    'bytecode': guvectorizers,
    'ast':      ast_guvectorizers,
}

def GUVectorize(func, signature, backend='ast', target='cpu'):
    assert backend in guvectorizers_backends
    targets = guvectorizers_backends[backend]
    assert target in targets
    return targets[target](func, signature)

def _prepare_sig(sig):
    if isinstance(sig, str):
        _name, restype, argtypes = _process_sig(str(sig), None)
    else:
        argtypes = sig.args
        restype = sig.return_type

    kws = {}
    if restype is not None:
        kws['restype'] = restype
    if argtypes is not None:
        kws['argtypes'] = argtypes

    return kws

def vectorize(signatures, backend='ast', target='cpu'):
    def _vectorize(fn):
        vect = Vectorize(fn, backend=backend, target=target)
        for sig in signatures:
            kws = _prepare_sig(sig)
            vect.add(**kws)
        ufunc = vect.build_ufunc()
        return ufunc

    return _vectorize

def guvectorize(fnsigs, gusig, backend='ast', target='cpu'):
    def _guvectorize(fn):
        vect = GUVectorize(fn, gusig, backend=backend, target=target)
        for sig in fnsigs:
            kws = _prepare_sig(sig)
            vect.add(**kws)
        ufunc = vect.build_ufunc()
        return ufunc

    return _guvectorize
