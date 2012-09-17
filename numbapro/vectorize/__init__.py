from .basic import BasicVectorize
from .parallel import ParallelVectorize
from .stream import StreamVectorize

try:
    from .cuda import  CudaVectorize
except ImportError, e:
    CudaVectorize = BasicVectorize

from .minivectorize import MiniVectorize, ParallelMiniVectorize
from .gufunc import GUFuncVectorize