import copy
import ctypes
import numpy
from numbapro.npm import types
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
    def __init__(self, name, ptx, argtys, excs):
        super(CUDAKernel, self).__init__()
        self.name = name                # to lookup entry kernel
        self.ptx = ptx                  # for debug and inspection
        self.excs = excs                # exception table
        self.linkfiles = []             # external files to be linked
        self.argtys = argtys
        self.c_argtys = [t.ctype_as_argument() for t in argtys]

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
        has_autotune = (hasattr(self, '_autotune') and
                        self._autotune.dynsmem == self.sharedmem)
        if has_autotune:
            return self._autotune
        else:
            at = AutoTuner.parse(self.name, self.compile_info,
                                 cc=self.device.COMPUTE_CAPABILITY)
            if at is None:
                raise RuntimeError('driver does not report compiliation info')
            self._autotune = at
            return self._autotune

    @property
    def occupancy(self):
        '''calculate the theoretical occupancy of the kernel given the
        configuation.
        '''
        return self.autotune.closest(self.thread_per_block)

    @property
    def thread_per_block(self):
        tpb = 1
        for b in self.blockdim:
            tpb *= b
        return tpb

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
                for t, v in zip(self.argtys, args)]

        # allocate space for exception
        if self.excs:
            excsize = ctypes.sizeof(ctypes.c_uint32)
            excmem = driver.DeviceMemory(excsize)
            driver.device_memset(excmem, 0, excsize)
            args.append(excmem.device_ctypes_pointer)

        # configure kernel
        cu_func = self.cu_function.configure(griddim, blockdim,
                                             stream=stream,
                                             sharedmem=sharedmem)
        # invoke kernel
        cu_func(*args)

        # check exceptions
        if self.excs:
            exchost = ctypes.c_uint32(0xffffffff)
            exc = driver.device_to_host(ctypes.addressof(exchost), excmem,
                                        excsize)
            errcode = exchost.value
            if errcode != 0:
                raise self.excs[errcode]

        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr):
        if isinstance(ty.desc, types.Array):
            devary, conv = devicearray.auto_device(val, stream=stream)
            if conv:
                retr.append(lambda: devary.copy_to_host(val, stream=stream))
            return devary.as_cuda_arg()
        elif isinstance(ty.desc, types.Complex):
            size = ctypes.sizeof(ty.desc.ctype_value())
            dmem = driver.DeviceMemory(size)
            cval = ty.desc.ctype_value()(val)
            driver.host_to_device(dmem, ctypes.addressof(cval), size,
                                  stream=stream)
            return dmem
        else:
            return ty.ctype_pack_argument(val)
