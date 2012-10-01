__all__ = [
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
    'basic': BasicVectorize,
    'parallel': ParallelVectorize,
    'stream': StreamVectorize,
    'gpu': CudaVectorize,
}

ast_vectorizers = {
    'basic': BasicASTVectorize,
    'parallel': ParallelASTVectorize,
    'stream': StreamASTVectorize,
}

mini_vectorizers = {
    'basic': MiniVectorize,
    'parallel': ParallelMiniVectorize,
}

backends = {
    'bytecode': vectorizers,
    'ast': ast_vectorizers,
    'mini': mini_vectorizers,
}

def Vectorize(func, backend='bytecode', target='basic'):
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
    'basic': GUFuncVectorize,
    'gpu': CudaGUFuncVectorize,
}

ast_guvectorizers = {
    'basic': GUFuncASTVectorize,
    'gpu': CudaGUFuncASTVectorize,
}

def GUVectorize(func, signature, backend='bytecode', target='basic'):
    assert backend in ('bytecode', 'ast')
    assert target in ('basic', 'gpu')

    if backend == 'bytecode':
        vs = guvectorizers
    else:
        vs = ast_guvectorizers

    return vs[target](func, signature)
