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
    'gpu': CudaVectorize,
}

ast_vectorizers = {
    'cpu': BasicASTVectorize,
    'parallel': ParallelASTVectorize,
    'stream': StreamASTVectorize,
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
    assert target in vectorizers, tuple(vectorizers)

    if target in backends[backend]:
        return backends[backend][target](func)
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

def GUVectorize(func, signature, backend='ast', target='cpu'):
    assert backend in ('bytecode', 'ast')
    assert target in ('cpu', 'gpu')

    if backend == 'bytecode':
        vs = guvectorizers
    else:
        vs = ast_guvectorizers

    return vs[target](func, signature)
