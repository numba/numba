from __future__ import absolute_import, print_function
import copy
from numba import compiler, types
from numba.typing.templates import ConcreteTemplate
from numbapro.cudadrv.devices import get_context
from numbapro.cudadrv import nvvm, devicearray


def compile_cuda(pyfunc, return_type, args, debug):
    # First compilation will trigger the initialization of the CUDA backend.
    from .descriptor import CUDATargetDesc

    typingctx = CUDATargetDesc.typingctx
    targetctx = CUDATargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})

    # Linking depending libraries
    targetctx.link_dependencies(cres.llvm_module, cres.target_context.linking)
    return cres


def compile_kernel(pyfunc, args, link, debug=False):
    cres = compile_cuda(pyfunc, types.void, args, debug=debug)
    kernel = cres.target_context.prepare_cuda_kernel(cres.llvm_func,
                                                     cres.signature.args)
    cres = cres._replace(llvm_func=kernel)
    cukern = CUDAKernel(llvm_module=cres.llvm_module,
                        name=cres.llvm_func.name,
                        argtypes=cres.signature.args,
                        link=link)
    return cukern


def compile_device(pyfunc, return_type, args, inline=True, debug=False):
    cres = compile_cuda(pyfunc, return_type, args, debug=debug)
    devfn = DeviceFunction(cres)

    class device_function_template(ConcreteTemplate):
        key = devfn
        cases = [cres.signature]

    cres.typing_context.insert_user_function(devfn, device_function_template)
    libs = [cres.llvm_module]
    cres.target_context.insert_user_function(devfn, cres.fndesc, libs)
    return devfn


class DeviceFunction(object):
    def __init__(self, cres):
        self.cres = cres


class CUDARuntimeError(RuntimeError):
    def __init__(self, exc, tx, ty, tz, bx, by):
        self.tid = tx, ty, tz
        self.ctaid = bx, by
        self.exc = exc
        t = ("An exception was raised in thread=%s block=%s\n"
             "\t%s: %s")
        msg = t % (self.tid, self.ctaid, type(self.exc).__name__, self.exc)
        super(CUDARuntimeError, self).__init__(msg)


class CUDAKernelBase(object):
    """Define interface for configurable kernels
    """

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


class CachedPTX(object):
    """A PTX cache that uses compute capability as a cache key
    """

    def __init__(self, llvmir):
        self.llvmir = llvmir
        self.cache = {}

    def get(self):
        """
        Get PTX for the current active context.
        """
        cuctx = get_context()
        device = cuctx.device
        cc = device.compute_capability
        ptx = self.cache.get(cc)
        if ptx is None:
            arch = nvvm.get_arch_option(*cc)
            ptx = nvvm.llvm_to_ptx(self.llvmir, opt=3, arch=arch)
            self.cache[cc] = ptx
        return ptx


class CachedCUFunction(object):
    """
    Get or compile CUDA function for the current active context

    Uses device ID as key for cache.
    """

    def __init__(self, entry_name, ptx):
        self.entry_name = entry_name
        self.ptx = ptx
        self.cache = {}

    def get(self):
        cuctx = get_context()
        device = cuctx.device
        cufunc = self.cache.get(device.id)
        if cufunc is None:
            ptx = self.ptx.get()
            module = cuctx.create_module_ptx(ptx)
            cufunc = module.get_function(self.entry_name)
            self.cache[device.id] = cufunc
        return cufunc


class CUDAKernel(CUDAKernelBase):
    def __init__(self, llvm_module, name, argtypes, link=()):
        super(CUDAKernel, self).__init__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self._linking = tuple(link)
        ptx = CachedPTX(str(llvm_module))
        self._func = CachedCUFunction(self.entry_name, ptx)

    def __call__(self, *args, **kwargs):
        assert not kwargs
        self._kernel_call(args=args,
                          griddim=self.griddim,
                          blockdim=self.blockdim,
                          stream=self.stream,
                          sharedmem=self.sharedmem)

    def bind(self):
        """
        Force binding to current CUDA context
        """
        self._func.get()

    def _kernel_call(self, args, griddim, blockdim, stream=0, sharedmem=0):
        # Prepare kernel
        cufunc = self._func.get()
        # Prepare arguments
        retr = []                       # hold functors for writeback
        args = [self._prepare_args(t, v, stream, retr)
                for t, v in zip(self.argument_types, args)]

        # XXX
        # # allocate space for exception
        # if self.excs:
        #     ctx = get_context()
        #     excsize = ctypes.sizeof(ctypes.c_int32) * 6
        #     excmem = ctx.memalloc(excsize)
        #     driver.device_memset(excmem, 0, excsize)
        #     args.append(excmem.device_ctypes_pointer)

        # Configure kernel
        cu_func = cufunc.configure(griddim, blockdim,
                                   stream=stream,
                                   sharedmem=sharedmem)
        # invoke kernel
        cu_func(*args)

        # XXX
        # # check exceptions
        # if self.excs:
        #     exchost = (ctypes.c_int32 * 6)()
        #     driver.device_to_host(ctypes.addressof(exchost), excmem,
        #                           excsize)
        #     if exchost[0] != 0:
        #         raise CUDARuntimeError(self.excs[exchost[0]].exc, *exchost[1:])

        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr):
        if isinstance(ty, types.Array):
            devary, conv = devicearray.auto_device(val, stream=stream)
            if conv:
                retr.append(lambda: devary.copy_to_host(val, stream=stream))
            return devary.as_cuda_arg()
        elif isinstance(ty, types.Complex):
            raise NotImplementedError
            ctx = get_context()
            size = ctypes.sizeof(ty.desc.ctype_value())
            dmem = ctx.memalloc(size)
            cval = ty.desc.ctype_value()(val)
            driver.host_to_device(dmem, ctypes.addressof(cval), size,
                                  stream=stream)
            return dmem
        else:
            raise NotImplementedError
            return ty.ctype_pack_argument(val)


