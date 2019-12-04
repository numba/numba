from __future__ import absolute_import, print_function


import os
from functools import reduce, wraps
import operator
import sys
import threading
import warnings

import numpy as np

from numba import ctypes_support as ctypes
from numba import config, compiler, types, sigutils
from numba.typing.templates import AbstractTemplate, ConcreteTemplate
from numba import funcdesc, typing, utils, serialize
from numba.compiler_lock import global_compiler_lock

from .cudadrv.autotune import AutoTuner
from .cudadrv.devices import get_context
from .cudadrv import nvvm, devicearray, driver
from .errors import normalize_kernel_dimensions
from .api import get_current_device
from .args import wrap_arg


@global_compiler_lock
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
        flags.set('debuginfo')
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


@global_compiler_lock
def compile_kernel(pyfunc, args, link, debug=False, inline=False,
                   fastmath=False, extensions=[], max_registers=None):
    cres = compile_cuda(pyfunc, types.void, args, debug=debug, inline=inline)
    fname = cres.fndesc.llvm_func_name
    lib, kernel = cres.target_context.prepare_cuda_kernel(cres.library, fname,
                                                          cres.signature.args,
                                                          debug=debug)

    cukern = CUDAKernel(llvm_module=lib._final_module,
                        name=kernel.name,
                        pretty_name=cres.fndesc.qualname,
                        argtypes=cres.signature.args,
                        type_annotation=cres.type_annotation,
                        link=link,
                        debug=debug,
                        call_helper=cres.call_helper,
                        fastmath=fastmath,
                        extensions=extensions,
                        max_registers=max_registers)
    return cukern


class DeviceFunctionTemplate(object):
    """Unmaterialized device function
    """
    def __init__(self, pyfunc, debug, inline):
        self.py_func = pyfunc
        self.debug = debug
        self.inline = inline
        self._compileinfos = {}

    def __reduce__(self):
        glbls = serialize._get_function_globals_for_reduction(self.py_func)
        func_reduced = serialize._reduce_function(self.py_func, glbls)
        args = (self.__class__, func_reduced, self.debug, self.inline)
        return (serialize._rebuild_reduction, args)

    @classmethod
    def _rebuild(cls, func_reduced, debug, inline):
        func = serialize._rebuild_function(*func_reduced)
        return compile_device_template(func, debug=debug, inline=inline)

    def compile(self, args):
        """Compile the function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
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

        return cres

    def inspect_llvm(self, args):
        """Returns the LLVM-IR text compiled for *args*.

        Parameters
        ----------
        args: tuple[Type]
            Argument types.

        Returns
        -------
        llvmir : str
        """
        cres = self._compileinfos[args]
        mod = cres.library._final_module
        return str(mod)

    def inspect_ptx(self, args, nvvm_options={}):
        """Returns the PTX compiled for *args* for the currently active GPU

        Parameters
        ----------
        args: tuple[Type]
            Argument types.
        nvvm_options : dict; optional
            See `CompilationUnit.compile` in `numba/cuda/cudadrv/nvvm.py`.

        Returns
        -------
        ptx : bytes
        """
        llvmir = self.inspect_llvm(args)
        # Make PTX
        cuctx = get_context()
        device = cuctx.device
        cc = device.compute_capability
        arch = nvvm.get_arch_option(*cc)
        ptx = nvvm.llvm_to_ptx(llvmir, opt=3, arch=arch, **nvvm_options)
        return ptx


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
            return dft.compile(args).signature

    typingctx = CUDATargetDesc.typingctx
    typingctx.insert_user_function(dft, device_function_template)
    return dft


def compile_device(pyfunc, return_type, args, inline=True, debug=False):
    return DeviceFunction(pyfunc, return_type, args, inline=True, debug=False)


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

    def __init__(self, pyfunc, return_type, args, inline, debug):
        self.py_func = pyfunc
        self.return_type = return_type
        self.args = args
        self.inline = True
        self.debug = False
        cres = compile_cuda(self.py_func, self.return_type, self.args,
                            debug=self.debug, inline=self.inline)
        self.cres = cres
        # Register
        class device_function_template(ConcreteTemplate):
            key = self
            cases = [cres.signature]

        cres.typing_context.insert_user_function(
            self, device_function_template)
        cres.target_context.insert_user_function(self, cres.fndesc,
                                                 [cres.library])

    def __reduce__(self):
        globs = serialize._get_function_globals_for_reduction(self.py_func)
        func_reduced = serialize._reduce_function(self.py_func, globs)
        args = (self.__class__, func_reduced, self.return_type, self.args,
                self.inline, self.debug)
        return (serialize._rebuild_reduction, args)

    @classmethod
    def _rebuild(cls, func_reduced, return_type, args, inline, debug):
        return cls(serialize._rebuild_function(*func_reduced), return_type,
                   args, inline, debug)

    def __repr__(self):
        fmt = "<DeviceFunction py_func={0} signature={1}>"
        return fmt.format(self.py_func, self.cres.signature)


class ExternFunction(object):
    def __init__(self, name, sig):
        self.name = name
        self.sig = sig


class ForAll(object):
    def __init__(self, kernel, ntasks, tpb, stream, sharedmem):
        self.kernel = kernel
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if isinstance(self.kernel, AutoJitCUDAKernel):
            kernel = self.kernel.specialize(*args)
        else:
            kernel = self.kernel

        tpb = self._compute_thread_per_block(kernel)
        tpbm1 = tpb - 1
        blkct = (self.ntasks + tpbm1) // tpb

        return kernel.configure(blkct, tpb, stream=self.stream,
                                sharedmem=self.sharedmem)(*args)

    def _compute_thread_per_block(self, kernel):
        tpb = self.thread_per_block
        # Prefer user-specified config
        if tpb != 0:
            return tpb
        # Else, ask the driver to give a good cofnig
        else:
            ctx = get_context()
            kwargs = dict(
                func=kernel._func.get(),
                b2d_func=0,     # dynamic-shared memory is constant to blksz
                memsize=self.sharedmem,
                blocksizelimit=1024,
            )
            try:
                # Raises from the driver if the feature is unavailable
                _, tpb = ctx.get_max_potential_block_size(**kwargs)
            except AttributeError:
                # Fallback to table-based approach.
                tpb = self._fallback_autotune_best(kernel)
                raise
            return tpb

    def _fallback_autotune_best(self, kernel):
        try:
            tpb = kernel.autotune.best()
        except ValueError:
            warnings.warn('Could not autotune, using default tpb of 128')
            tpb = 128

        return tpb


class CUDAKernelBase(object):
    """Define interface for configurable kernels
    """

    def __init__(self):
        self.griddim = (1, 1)
        self.blockdim = (1, 1, 1)
        self.sharedmem = 0
        self.stream = 0

    def copy(self):
        """
        Shallow copy the instance
        """
        # Note: avoid using ``copy`` which calls __reduce__
        cls = self.__class__
        # new bare instance
        new = cls.__new__(cls)
        # update the internal states
        new.__dict__.update(self.__dict__)
        return new

    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)

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

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        """Returns a configured kernel for 1D kernel of given number of tasks
        ``ntasks``.

        This assumes that:
        - the kernel 1-to-1 maps global thread id ``cuda.grid(1)`` to tasks.
        - the kernel must check if the thread id is valid."""

        return ForAll(self, ntasks, tpb=tpb, stream=stream, sharedmem=sharedmem)

    def _serialize_config(self):
        """
        Helper for serializing the grid, block and shared memory configuration.
        CUDA stream config is not serialized.
        """
        return self.griddim, self.blockdim, self.sharedmem

    def _deserialize_config(self, config):
        """
        Helper for deserializing the grid, block and shared memory
        configuration.
        """
        self.griddim, self.blockdim, self.sharedmem = config


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

    def __init__(self, entry_name, ptx, linking, max_registers):
        self.entry_name = entry_name
        self.ptx = ptx
        self.linking = linking
        self.cache = {}
        self.ccinfos = {}
        self.max_registers = max_registers

    def get(self):
        cuctx = get_context()
        device = cuctx.device
        cufunc = self.cache.get(device.id)
        if cufunc is None:
            ptx = self.ptx.get()

            # Link
            linker = driver.Linker(max_registers=self.max_registers)
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

    def __reduce__(self):
        """
        Reduce the instance for serialization.
        Pre-compiled PTX code string is serialized inside the `ptx` (CachedPTX).
        Loaded CUfunctions are discarded. They are recreated when unserialized.
        """
        if self.linking:
            msg = ('cannot pickle CUDA kernel function with additional '
                   'libraries to link against')
            raise RuntimeError(msg)
        args = (self.__class__, self.entry_name, self.ptx, self.linking, self.max_registers)
        return (serialize._rebuild_reduction, args)

    @classmethod
    def _rebuild(cls, entry_name, ptx, linking, max_registers):
        """
        Rebuild an instance.
        """
        return cls(entry_name, ptx, linking, max_registers)


class CUDAKernel(CUDAKernelBase):
    '''
    CUDA Kernel specialized for a given set of argument types. When called, this
    object will validate that the argument types match those for which it is
    specialized, and then launch the kernel on the device.
    '''
    def __init__(self, llvm_module, name, pretty_name, argtypes, call_helper,
                 link=(), debug=False, fastmath=False, type_annotation=None,
                 extensions=[], max_registers=None):
        super(CUDAKernel, self).__init__()
        # initialize CUfunction
        options = {'debug': debug}
        if fastmath:
            options.update(dict(ftz=True,
                                prec_sqrt=False,
                                prec_div=False,
                                fma=True))

        ptx = CachedPTX(pretty_name, str(llvm_module), options=options)
        cufunc = CachedCUFunction(name, ptx, link, max_registers)
        # populate members
        self.entry_name = name
        self.argument_types = tuple(argtypes)
        self.linking = tuple(link)
        self._type_annotation = type_annotation
        self._func = cufunc
        self.debug = debug
        self.call_helper = call_helper
        self.extensions = list(extensions)

    @classmethod
    def _rebuild(cls, name, argtypes, cufunc, link, debug, call_helper, extensions, config):
        """
        Rebuild an instance.
        """
        instance = cls.__new__(cls)
        # invoke parent constructor
        super(cls, instance).__init__()
        # populate members
        instance.entry_name = name
        instance.argument_types = tuple(argtypes)
        instance.linking = tuple(link)
        instance._type_annotation = None
        instance._func = cufunc
        instance.debug = debug
        instance.call_helper = call_helper
        instance.extensions = extensions
        # update config
        instance._deserialize_config(config)
        return instance

    def __reduce__(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are serialized in PTX form.
        Type annotation are discarded.
        Thread, block and shared memory configuration are serialized.
        Stream information is discarded.
        """
        config = self._serialize_config()
        args = (self.__class__, self.entry_name, self.argument_types,
                self._func, self.linking, self.debug, self.call_helper,
                self.extensions, config)
        return (serialize._rebuild_reduction, args)

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
        return self._func.ptx.get().decode('ascii')

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
                exccls, exc_args, loc = self.call_helper.get_exception(code)
                # Prefix the exception message with the source location
                if loc is None:
                    locinfo = ''
                else:
                    sym, filepath, lineno = loc
                    filepath = os.path.relpath(filepath)
                    locinfo = 'In function %r, file %s, line %s, ' % (
                        sym, filepath, lineno,
                        )
                # Prefix the exception message with the thread position
                prefix = "%stid=%s ctaid=%s" % (locinfo, tid, ctaid)
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

        # map the arguments using any extension you've registered
        for extension in reversed(self.extensions):
            ty, val = extension.prepare_args(
                ty,
                val,
                stream=stream,
                retr=retr)

        if isinstance(ty, types.Array):
            devary = wrap_arg(val).to_device(retr, stream)

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

        elif isinstance(ty, (types.NPDatetime, types.NPTimedelta)):
            kernelargs.append(ctypes.c_int64(val.view(np.int64)))

        elif isinstance(ty, types.Record):
            devrec = wrap_arg(val).to_device(retr, stream)
            kernelargs.append(devrec)

        else:
            raise NotImplementedError(ty, val)

    @property
    def autotune(self):
        """Return the autotuner object associated with this kernel."""
        warnings.warn(_deprec_warn_msg.format('autotune'), DeprecationWarning)
        has_autotune = hasattr(self, '_autotune')
        if has_autotune and self._autotune.dynsmem == self.sharedmem:
            return self._autotune
        else:
            # Get CUDA Function
            cufunc = self._func.get()
            at = AutoTuner(info=cufunc.attrs, cc=cufunc.device.compute_capability)
            self._autotune = at
            return self._autotune

    @property
    def occupancy(self):
        """Occupancy is the ratio of the number of active warps per multiprocessor to the maximum
        number of warps that can be active on the multiprocessor at once.
        Calculate the theoretical occupancy of the kernel given the
        current configuration."""
        warnings.warn(_deprec_warn_msg.format('occupancy'), DeprecationWarning)
        thread_per_block = reduce(operator.mul, self.blockdim, 1)
        return self.autotune.closest(thread_per_block)


_deprec_warn_msg = ("The .{} attribute is is deprecated and will be "
                    "removed in a future release")


class AutoJitCUDAKernel(CUDAKernelBase):
    '''
    CUDA Kernel object. When called, the kernel object will specialize itself
    for the given arguments (if no suitable specialized version already exists)
    & compute capability, and launch on the device associated with the current
    context.

    Kernel objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    '''
    def __init__(self, func, bind, targetoptions):
        super(AutoJitCUDAKernel, self).__init__()
        self.py_func = func
        self.bind = bind

        # keyed by a `(compute capability, args)` tuple
        self.definitions = {}

        self.targetoptions = targetoptions

        # defensive copy
        self.targetoptions['extensions'] = \
            list(self.targetoptions.get('extensions', []))

        from .descriptor import CUDATargetDesc

        self.typingctx = CUDATargetDesc.typingctx

    @property
    def extensions(self):
        '''
        A list of objects that must have a `prepare_args` function. When a
        specialized kernel is called, each argument will be passed through
        to the `prepare_args` (from the last object in this list to the
        first). The arguments to `prepare_args` are:

        - `ty` the numba type of the argument
        - `val` the argument value itself
        - `stream` the CUDA stream used for the current call to the kernel
        - `retr` a list of zero-arg functions that you may want to append
          post-call cleanup work to.

        The `prepare_args` function must return a tuple `(ty, val)`, which
        will be passed in turn to the next right-most `extension`. After all
        the extensions have been called, the resulting `(ty, val)` will be
        passed into Numba's default argument marshalling logic.
        '''
        return self.targetoptions['extensions']

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
        kernel = self.compile(argtypes)
        return kernel

    def compile(self, sig):
        '''
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        '''
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None
        cc = get_current_device().compute_capability
        kernel = self.definitions.get((cc, argtypes))
        if kernel is None:
            if 'link' not in self.targetoptions:
                self.targetoptions['link'] = ()
            kernel = compile_kernel(self.py_func, argtypes,
                                    **self.targetoptions)
            self.definitions[(cc, argtypes)] = kernel
            if self.bind:
                kernel.bind()
        return kernel

    def inspect_llvm(self, signature=None, compute_capability=None):
        '''
        Return the LLVM IR for all signatures encountered thus far, or the LLVM
        IR for a specific signature and compute_capability if given.
        '''
        cc = compute_capability or get_current_device().compute_capability
        if signature is not None:
            return self.definitions[(cc, signature)].inspect_llvm()
        else:
            return dict((sig, defn.inspect_llvm())
                        for sig, defn in self.definitions.items())

    def inspect_asm(self, signature=None, compute_capability=None):
        '''
        Return the generated assembly code for all signatures encountered thus
        far, or the LLVM IR for a specific signature and compute_capability
        if given.
        '''
        cc = compute_capability or get_current_device().compute_capability
        if signature is not None:
            return self.definitions[(cc, signature)].inspect_asm()
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

        for _, defn in utils.iteritems(self.definitions):
            defn.inspect_types(file=file)

    @classmethod
    def _rebuild(cls, func_reduced, bind, targetoptions, config):
        """
        Rebuild an instance.
        """
        func = serialize._rebuild_function(*func_reduced)
        instance = cls(func, bind, targetoptions)
        instance._deserialize_config(config)
        return instance

    def __reduce__(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are discarded.
        """
        glbls = serialize._get_function_globals_for_reduction(self.py_func)
        func_reduced = serialize._reduce_function(self.py_func, glbls)
        config = self._serialize_config()
        args = (self.__class__, func_reduced, self.bind, self.targetoptions,
                config)
        return (serialize._rebuild_reduction, args)
