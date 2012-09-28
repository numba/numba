from .basic import BasicVectorize
from .parallel import ParallelVectorize
from .stream import StreamVectorize

from .gufunc import GUFuncVectorize, ASTGUFuncVectorize

try:
    from .cuda import  CudaVectorize
    from .gufunc import CUDAGUFuncVectorize
except ImportError, e:
    CudaVectorize = BasicVectorize
    CUDAGUFuncVectorize = GUFuncVectorize

from .minivectorize import MiniVectorize, ParallelMiniVectorize