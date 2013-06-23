import copy
from numbapro.npm.execution import (to_ctype, prepare_args, Complex64,
                                    Complex128, ArrayBase)
from numbapro.cudadrv import driver, devicearray

class CUDAKernelBase(object):
    '''Define interface for configurable kernels
    '''
    def __init__(self):
        self.griddim = (1, 1)
        self.blockdim = (1, 1, 1)
        self.sharedmem = 0
        self.stream = 0

    def copy(self):
        return copy.copy(self)

    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        if not isinstance(griddim, (tuple, list)):
            griddim = [griddim]
        else:
            griddim = list(griddim)
        if len(griddim) > 2:
            raise ValueError('griddim must be a tuple/list of two ints')
        while len(griddim) < 2:
            griddim.append(1)

        if not isinstance(blockdim, (tuple, list)):
            blockdim = [blockdim]
        else:
            blockdim = list(blockdim)
        if len(blockdim) > 3:
            raise ValueError('blockdim must be tuple/list of three ints')
        while len(blockdim) < 3:
            blockdim.append(1)

        clone = self.copy()
        clone.griddim = tuple(griddim)
        clone.blockdim = tuple(blockdim)
        clone.stream = stream
        clone.sharedmem = sharedmem
        return clone

    def __getitem__(self, args):
        if len(args) not in [2, 3, 4]:
            raise ValueError('must specify at least the griddim and blockdim')
        return self.configure(*args)



class CUDAKernel(CUDAKernelBase):
    '''A callable object representing a CUDA kernel.
    '''
    def __init__(self, name, ptx, argtys):
        super(CUDAKernel, self).__init__()
        self.name = name                # to lookup entry kernel
        self.ptx = ptx                  # for debug and inspection
        self.c_argtys = [to_ctype(t) for t in argtys]

    def bind(self):
        '''Associate to the current context.
        NOTE: free to invoke for multiple times.
        '''
        self.cu_module = driver.Module(self.ptx)
        self.cu_function = driver.Function(self.cu_module, self.name)

    @property
    def device(self):
        return self.cu_function.device

    def __call__(self, *args):
        self._call(args = args,
                   griddim = self.griddim,
                   blockdim = self.blockdim,
                   stream = self.stream,
                   sharedmem = self.sharedmem)

    def _call(self, args, griddim, blockdim, stream=0, sharedmem=0):
        # prepare arguments
        retr = []                       # hold functors for writeback
        args = [self._prepare_args(t, v, stream, retr)
                for t, v in zip(self.c_argtys, args)]
        # configure kernel
        cu_func = self.cu_function.configure(griddim, blockdim,
                                             stream=stream,
                                             sharedmem=sharedmem)
        # invoke kernel
        cu_func(*args)
        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr):
        if issubclass(ty, ArrayBase):
            devary, conv = devicearray.auto_device(val, stream=stream)
            if conv:
                retr.append(lambda: devary.copy_to_host(val, stream=stream))
            return devary.as_cuda_arg()
        elif issubclass(ty, (Complex64, Complex128)):
            raise NotImplementedError("complex argument is not supported")
        else:
            return prepare_args(ty, val)
