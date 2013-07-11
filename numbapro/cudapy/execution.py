import copy
import ctypes
from numbapro.npm.execution import (to_ctype, prepare_args, Complex64,
                                    Complex128, ArrayBase)
from numbapro.cudadrv import driver, devicearray
from numbapro.cudadrv.autotune import AutoTuner

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
        self.linkfiles = []             # external files to be linked
        self.c_argtys = [to_ctype(t) for t in argtys]

    def bind(self):
        '''Associate to the current context.
        NOTE: free to invoke for multiple times.
        '''
        if self.linkfiles:
            self.link()
        else:
            self.cu_module = driver.Module(ptx=self.ptx)
            self.compile_info = self.cu_module.info_log

        self.cu_function = driver.Function(self.cu_module, self.name)

    def link(self):
        '''Link external files
        '''
        linker = driver.Linker()
        linker.add_ptx(self.ptx)
        for file in self.linkfiles:
            linker.add_file_guess_ext(file)
        cubin, _size = linker.complete()
        self.cu_module = driver.Module(image=cubin)
        self.compile_info = linker.info_log

    @property
    def device(self):
        return self.cu_function.device

    @property
    def autotune(self):
        try:
            return self._autotune
        except AttributeError:
            self._autotune = AutoTuner(self.name, self.compile_info,
                                       cc=self.device.COMPUTE_CAPABILITY)
            return self._autotune

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
            size = ctypes.sizeof(ty)
            dmem = driver.DeviceMemory(size)
            cval = ty(val)
            driver.host_to_device(dmem, ctypes.addressof(cval), size,
                                  stream=stream)
            return dmem
        else:
            return prepare_args(ty, val)
