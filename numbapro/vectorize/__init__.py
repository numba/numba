__all__ = [
    'Vectorize',
    'BasicVectorize',
    'ParallelVectorize',
    'StreamVectorize',
    'GUFuncVectorize',
    'GUFuncASTVectorize',
    'CudaVectorize',
    'CudaGUFuncVectorize',
    'MiniVectorize',
    'ParallelMiniVectorize',
]

from .basic import BasicVectorize, BasicASTVectorize
from .parallel import ParallelVectorize, ParallelASTVectorize
from .stream import StreamVectorize, StreamASTVectorize

from .gufunc import GUFuncVectorize, GUFuncASTVectorize

try:
    from .cuda import  CudaVectorize
    from .gufunc import CudaGUFuncVectorize, CudaGUFuncASTVectorize
except ImportError, e:
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
    'gpu': CudaVectorize,
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
    'gpu': CudaGUFuncVectorize,
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
