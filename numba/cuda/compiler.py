from __future__ import absolute_import, print_function
import copy
import ctypes
from numba import compiler, types, errcode
from numba.typing.templates import ConcreteTemplate
from numba import typing, lowering, dispatcher
from .cudadrv.devices import get_context
from .cudadrv import nvvm, devicearray, driver
from .errors import KernelRuntimeError
from .api import get_current_device


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

    # Fix global naming
    for gv in cres.llvm_module.global_variables:
        if '.' in gv.name:
            gv.name = gv.name.replace('.', '_')

    return cres


def compile_kernel(pyfunc, args, link, debug=False):
    cres = compile_cuda(pyfunc, types.void, args, debug=debug)
    kernel = cres.target_context.prepare_cuda_kernel(cres.llvm_func,
                                                     cres.signature.args)
    cres = cres._replace(llvm_func=kernel)
    cukern = CUDAKernel(llvm_module=cres.llvm_module,
                        name=cres.llvm_func.name,
                        argtypes=cres.signature.args,
                        link=link,
                        debug=debug,
                        exceptions=cres.target_context.exceptions)
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


def declare_device_function(name, restype, argtypes):
    from .descriptor import CUDATargetDesc

    typingctx = CUDATargetDesc.typingctx
    targetctx = CUDATargetDesc.targetctx
    sig = typing.signature(restype, *argtypes)
    extfn = ExternFunction(name, sig)

    class device_function_template(ConcreteTemplate):
        key = extfn
        cases = [sig]

    fndesc = lowering.ExternalFunctionDescriptor(
        name=name, restype=restype, argtypes=argtypes)
    typingctx.insert_user_function(extfn, device_function_template)
    targetctx.insert_user_function(extfn, fndesc)
    return extfn


class DeviceFunction(object):
    def __init__(self, cres):
        self.cres = cres


class ExternFunction(object):
    def __init__(self, name, sig):
        self.name = name
        self.sig = sig


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

    def __init__(self, entry_name, ptx, linking):
        self.entry_name = entry_name
        self.ptx = ptx
        self.linking = linking
        self.cache = {}
        self.ccinfos = {}

    def get(self):
        cuctx = get_context()
        device = cuctx.device
        cufunc = self.cache.get(device.id)
        if cufunc is None:
            ptx = self.ptx.get()

            # Link
            linker = driver.Linker()
            linker.add_ptx(ptx)
            for path in self.linking:
                linker.add_file_guess_ext(path)
            cubin, _size = linker.complete()
            compile_info = linker.info_log
            module = cuctx.create_module_image(cubin)

            # Load
            cufunc = module.get_function(self.entry_name)
            self.cache[device.id] = cufunc
            self.ccinfos[device.id] = compile_info
        return cufunc

    def get_info(self):
        self.get()   # trigger compilation
        cuctx = get_context()
        device = cuctx.device
        ci = self.ccinfos[device.id]
        return ci


class CUDAKernel(CUDAKernelBase):
    def __init__(self, llvm_module, name, argtypes, link=(), debug=False,
                 exceptions={}):
        super(CUDAKernel, self).__init__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.linking = tuple(link)
        ptx = CachedPTX(str(llvm_module))
        self._func = CachedCUFunction(self.entry_name, ptx, link)
        self.debug = debug
        self.exceptions = exceptions

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

    @property
    def ptx(self):
        return self._func.ptx.get().decode('utf8')

    @property
    def device(self):
        """
        Get current active context
        """
        return get_current_device()

    def _kernel_call(self, args, griddim, blockdim, stream=0, sharedmem=0):
        # Prepare kernel
        cufunc = self._func.get()

        if self.debug:
            excname = cufunc.name + "__errcode__"
            excmem, excsz = cufunc.module.get_global_symbol(excname)
            assert excsz == ctypes.sizeof(ctypes.c_int)
            excval = ctypes.c_int()
            excmem.memset(0, stream=stream)

        # Prepare arguments
        retr = []                       # hold functors for writeback
        args = [self._prepare_args(t, v, stream, retr)
                for t, v in zip(self.argument_types, args)]

        # Configure kernel
        cu_func = cufunc.configure(griddim, blockdim,
                                   stream=stream,
                                   sharedmem=sharedmem)
        # invoke kernel
        cu_func(*args)

        if self.debug:
            driver.device_to_host(ctypes.addressof(excval), excmem, excsz)
            if excval.value != 0:
                # Error occurred
                def load_symbol(name):
                    mem, sz = cufunc.module.get_global_symbol("%s__%s__" %
                                                              (cufunc.name,
                                                               name))
                    val = ctypes.c_int()
                    driver.device_to_host(ctypes.addressof(val), mem, sz)
                    return val.value

                tid = [load_symbol("tid" + i) for i in 'zyx']
                ctaid = [load_symbol("ctaid" + i) for i in 'zyx']
                code = excval.value
                builtinerr = errcode.error_names.get(code)
                if builtinerr is not None:
                    raise KernelRuntimeError("code=%d reason=%s" %
                                             (code, builtinerr), tid=tid,
                                             ctaid=ctaid)
                else:
                    exccls = self.exceptions[code]
                    raise exccls("tid=%s ctaid=%s" % (tid, ctaid))

        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr):
        if isinstance(ty, types.Array):
            devary, conv = devicearray.auto_device(val, stream=stream)
            if conv:
                retr.append(lambda: devary.copy_to_host(val, stream=stream))
            return devary.as_cuda_arg()

        elif isinstance(ty, types.Integer):
            return getattr(ctypes, "c_%s" % ty)(val)

        elif ty == types.float64:
            return ctypes.c_double(val)

        elif ty == types.float32:
            return ctypes.c_float(val)

        elif ty == types.boolean:
            return ctypes.c_uint8(int(val))

        elif ty == types.complex64:
            ctx = get_context()
            size = ctypes.sizeof(Complex64)
            dmem = ctx.memalloc(size)
            cval = Complex64(val)
            driver.host_to_device(dmem, ctypes.addressof(cval), size,
                                  stream=stream)
            return dmem

        elif ty == types.complex128:
            ctx = get_context()
            size = ctypes.sizeof(Complex128)
            dmem = ctx.memalloc(size)
            cval = Complex128(val)
            driver.host_to_device(dmem, ctypes.addressof(cval), size,
                                  stream=stream)
            return dmem

        else:
            raise NotImplementedError(ty, val)


class Complex(ctypes.Structure):
    def __init__(self, val):
        super(Complex, self).__init__()
        if isinstance(val, complex):
            self.real = val.real
            self.imag = val.imag
        else:
            self.real = val


class Complex64(Complex):
    _fields_ = [
        ('real', ctypes.c_float),
        ('imag', ctypes.c_float)
    ]


class Complex128(Complex):
    _fields_ = [
        ('real', ctypes.c_double),
        ('imag', ctypes.c_double),
    ]


class AutoJitCUDAKernel(CUDAKernelBase):
    def __init__(self, func, bind, targetoptions):
        super(AutoJitCUDAKernel, self).__init__()
        self.py_func = func
        self.bind = bind
        self.definitions = {}
        self.targetoptions = targetoptions

    def __call__(self, *args):
        kernel = self.specialize(*args)
        cfg = kernel[self.griddim, self.blockdim, self.stream, self.sharedmem]
        cfg(*args)

    def specialize(self, *args):
        argtypes = tuple([dispatcher.Overloaded.typeof_pyval(a) for a in args])
        kernel = self.definitions.get(argtypes)
        if kernel is None:
            kernel = compile_kernel(self.py_func, argtypes, link=(),
                                    **self.targetoptions)
            self.definitions[argtypes] = kernel
            if self.bind:
                kernel.bind()
        return kernel
