from __future__ import absolute_import, print_function
import copy
import sys

from numba import ctypes_support as ctypes
from numba.typing.templates import AbstractTemplate
from numba import config, compiler, types
from numba.typing.templates import ConcreteTemplate
from numba import funcdesc, typing, utils

from .cudadrv.devices import get_context
from .cudadrv import nvvm, devicearray, driver
from .errors import KernelRuntimeError
from .api import get_current_device


def compile_cuda(pyfunc, return_type, args, debug, inline):
    # First compilation will trigger the initialization of the CUDA backend.
    from .descriptor import CUDATargetDesc

    typingctx = CUDATargetDesc.typingctx
    targetctx = CUDATargetDesc.targetctx
    # TODO handle debug flag
    flags = compiler.Flags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.set('no_compile')
    flags.set('no_cpython_wrapper')
    if debug:
        flags.set('boundcheck')
    if inline:
        flags.set('forceinline')
    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={})

    library = cres.library
    library.finalize()

    return cres


def compile_kernel(pyfunc, args, link, debug=False, inline=False,
                   fastmath=False):
    cres = compile_cuda(pyfunc, types.void, args, debug=debug, inline=inline)
    func = cres.library.get_function(cres.fndesc.llvm_func_name)
    kernel = cres.target_context.prepare_cuda_kernel(func,
                                                     cres.signature.args)
    cukern = CUDAKernel(llvm_module=cres.library._final_module,
                        name=kernel.name,
                        pretty_name=cres.fndesc.qualname,
                        argtypes=cres.signature.args,
                        type_annotation=cres.type_annotation,
                        link=link,
                        debug=debug,
                        call_helper=cres.call_helper,
                        fastmath=fastmath)
    return cukern


class DeviceFunctionTemplate(object):
    """Unmaterialized device function
    """
    def __init__(self, pyfunc, debug, inline):
        self.py_func = pyfunc
        self.debug = debug
        self.inline = inline
        self._compileinfos = {}

    def compile(self, args):
        """Compile the function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.
        """
        if args not in self._compileinfos:
            cres = compile_cuda(self.py_func, None, args, debug=self.debug,
                                inline=self.inline)
            first_definition = not self._compileinfos
            self._compileinfos[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(self, cres.fndesc,
                                                         libs)
            else:
                cres.target_context.add_user_function(self, cres.fndesc, libs)

        else:
            cres = self._compileinfos[args]

        return cres.signature


def compile_device_template(pyfunc, debug=False, inline=False):
    """Create a DeviceFunctionTemplate object and register the object to
    the CUDA typing context.
    """
    from .descriptor import CUDATargetDesc

    dft = DeviceFunctionTemplate(pyfunc, debug=debug, inline=inline)

    class device_function_template(AbstractTemplate):
        key = dft

        def generic(self, args, kws):
            assert not kws
            return dft.compile(args)

    typingctx = CUDATargetDesc.typingctx
    typingctx.insert_user_function(dft, device_function_template)
    return dft


def compile_device(pyfunc, return_type, args, inline=True, debug=False):
    cres = compile_cuda(pyfunc, return_type, args, debug=debug, inline=inline)
    devfn = DeviceFunction(cres)

    class device_function_template(ConcreteTemplate):
        key = devfn
        cases = [cres.signature]

    cres.typing_context.insert_user_function(devfn, device_function_template)
    cres.target_context.insert_user_function(devfn, cres.fndesc, [cres.library])
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

    fndesc = funcdesc.ExternalFunctionDescriptor(
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
        if len(griddim) > 3:
            raise ValueError('griddim must be a tuple/list of three ints')
        while len(griddim) < 3:
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
    def __init__(self, name, llvmir, options):
        self.name = name
        self.llvmir = llvmir
        self.cache = {}
        self._extra_options = options.copy()

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
            ptx = nvvm.llvm_to_ptx(self.llvmir, opt=3, arch=arch,
                                   **self._extra_options)
            self.cache[cc] = ptx
            if config.DUMP_ASSEMBLY:
                print(("ASSEMBLY %s" % self.name).center(80, '-'))
                print(ptx.decode('utf-8'))
                print('=' * 80)
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
    '''
    CUDA Kernel specialized for a given set of argument types. When called, this
    object will validate that the argument types match those for which it is
    specialized, and then launch the kernel on the device.
    '''
    def __init__(self, llvm_module, name, pretty_name,
                 argtypes, call_helper,
                 link=(), debug=False, fastmath=False,
                 type_annotation=None):
        super(CUDAKernel, self).__init__()
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.linking = tuple(link)
        self._type_annotation = type_annotation

        options = {}
        if fastmath:
            options.update(dict(ftz=True,
                                prec_sqrt=False,
                                prec_div=False,
                                fma=True))

        ptx = CachedPTX(pretty_name, str(llvm_module), options=options)
        self._func = CachedCUFunction(self.entry_name, ptx, link)
        self.debug = debug
        self.call_helper = call_helper

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
        '''
        PTX code for this kernel.
        '''
        return self._func.ptx.get().decode('utf8')

    @property
    def device(self):
        """
        Get current active context
        """
        return get_current_device()

    def inspect_llvm(self):
        '''
        Returns the LLVM IR for this kernel.
        '''
        return str(self._func.ptx.llvmir)

    def inspect_asm(self):
        '''
        Returns the PTX code for this kernel.
        '''
        return str(self._func.ptx.get())

    def inspect_types(self, file=None):
        '''
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        '''
        if self._type_annotation is None:
            raise ValueError("Type annotation is not available")

        if file is None:
            file = sys.stdout

        print("%s %s" % (self.entry_name, self.argument_types), file=file)
        print('-' * 80, file=file)
        print(self._type_annotation, file=file)
        print('=' * 80, file=file)

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

        kernelargs = []
        for t, v in zip(self.argument_types, args):
            self._prepare_args(t, v, stream, retr, kernelargs)

        # Configure kernel
        cu_func = cufunc.configure(griddim, blockdim,
                                   stream=stream,
                                   sharedmem=sharedmem)
        # Invoke kernel
        cu_func(*kernelargs)

        if self.debug:
            driver.device_to_host(ctypes.addressof(excval), excmem, excsz)
            if excval.value != 0:
                # An error occurred
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
                exccls, exc_args = self.call_helper.get_exception(code)
                # Prefix the exception message with the thread position
                prefix = "tid=%s ctaid=%s" % (tid, ctaid)
                if exc_args:
                    exc_args = ("%s: %s" % (prefix, exc_args[0]),) + exc_args[1:]
                else:
                    exc_args = prefix,
                raise exccls(*exc_args)

        # retrieve auto converted arrays
        for wb in retr:
            wb()

    def _prepare_args(self, ty, val, stream, retr, kernelargs):
        """
        Convert arguments to ctypes and append to kernelargs
        """
        if isinstance(ty, types.Array):
            devary, conv = devicearray.auto_device(val, stream=stream)
            if conv:
                retr.append(lambda: devary.copy_to_host(val, stream=stream))

            c_intp = ctypes.c_ssize_t

            meminfo = ctypes.c_void_p(0)
            parent = ctypes.c_void_p(0)
            nitems = c_intp(devary.size)
            itemsize = c_intp(devary.dtype.itemsize)
            data = ctypes.c_void_p(driver.device_pointer(devary))
            kernelargs.append(meminfo)
            kernelargs.append(parent)
            kernelargs.append(nitems)
            kernelargs.append(itemsize)
            kernelargs.append(data)
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.shape[ax]))
            for ax in range(devary.ndim):
                kernelargs.append(c_intp(devary.strides[ax]))

        elif isinstance(ty, types.Integer):
            cval = getattr(ctypes, "c_%s" % ty)(val)
            kernelargs.append(cval)

        elif ty == types.float64:
            cval = ctypes.c_double(val)
            kernelargs.append(cval)

        elif ty == types.float32:
            cval = ctypes.c_float(val)
            kernelargs.append(cval)

        elif ty == types.boolean:
            cval = ctypes.c_uint8(int(val))
            kernelargs.append(cval)

        elif ty == types.complex64:
            kernelargs.append(ctypes.c_float(val.real))
            kernelargs.append(ctypes.c_float(val.imag))

        elif ty == types.complex128:
            kernelargs.append(ctypes.c_double(val.real))
            kernelargs.append(ctypes.c_double(val.imag))

        elif isinstance(ty, types.Record):
            devrec, conv = devicearray.auto_device(val, stream=stream)
            if conv:
                retr.append(lambda: devrec.copy_to_host(val, stream=stream))

            kernelargs.append(devrec)

        else:
            raise NotImplementedError(ty, val)


class AutoJitCUDAKernel(CUDAKernelBase):
    '''
    CUDA Kernel object. When called, the kernel object will specialize itself
    for the given arguments (if no suitable specialized version already exists)
    and launch on the device associated with the current context.

    Kernel objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    '''
    def __init__(self, func, bind, targetoptions):
        super(AutoJitCUDAKernel, self).__init__()
        self.py_func = func
        self.bind = bind
        self.definitions = {}
        self.targetoptions = targetoptions

        from .descriptor import CUDATargetDesc

        self.typingctx = CUDATargetDesc.typingctx

    def __call__(self, *args):
        '''
        Specialize and invoke this kernel with *args*.
        '''
        kernel = self.specialize(*args)
        cfg = kernel[self.griddim, self.blockdim, self.stream, self.sharedmem]
        cfg(*args)

    def specialize(self, *args):
        '''
        Compile and bind to the current context a version of this kernel
        specialized for the given *args*.
        '''
        argtypes = tuple(
            [self.typingctx.resolve_argument_type(a) for a in args])
        kernel = self.definitions.get(argtypes)
        if kernel is None:
            if 'link' not in self.targetoptions:
                self.targetoptions['link'] = ()
            kernel = compile_kernel(self.py_func, argtypes,
                                    **self.targetoptions)
            self.definitions[argtypes] = kernel
            if self.bind:
                kernel.bind()
        return kernel

    def inspect_llvm(self, signature=None):
        '''
        Return the LLVM IR for all signatures encountered thus far, or the LLVM
        IR for a specific signature if given.
        '''
        if signature is not None:
            return self.definitions[signature].inspect_llvm()
        else:
            return dict((sig, defn.inspect_llvm())
                        for sig, defn in self.definitions.items())

    def inspect_asm(self, signature=None):
        '''
        Return the generated assembly code for all signatures encountered thus
        far, or the LLVM IR for a specific signature if given.
        '''
        if signature is not None:
            return self.definitions[signature].inspect_asm()
        else:
            return dict((sig, defn.inspect_asm())
                        for sig, defn in self.definitions.items())

    def inspect_types(self, file=None):
        '''
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        '''
        if file is None:
            file = sys.stdout

        for ver, defn in utils.iteritems(self.definitions):
            defn.inspect_types(file=file)
