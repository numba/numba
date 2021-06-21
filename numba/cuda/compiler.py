import collections
import ctypes
import functools
import inspect
import os
import sys

import numpy as np

from numba import _dispatcher
from numba.core.typing.templates import AbstractTemplate, ConcreteTemplate
from numba.core import (types, typing, utils, funcdesc, serialize, config,
                        compiler, sigutils)
from numba.core.typeconv.rules import default_type_manager
from numba.core.compiler import (CompilerBase, DefaultPassBuilder,
                                 compile_result, Flags, Option)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass, PassManager,
                                           register_pass)
from numba.core.dispatcher import OmittedArg
from numba.core.errors import NumbaDeprecationWarning
from numba.core.typed_passes import IRLegalization, NativeLowering
from numba.core.typing.typeof import Purpose, typeof
from warnings import warn
import numba
from .cudadrv.devices import get_context
from .cudadrv.libs import get_cudalib
from .cudadrv import driver
from .errors import missing_launch_config_msg, normalize_kernel_dimensions
from .api import get_current_device
from .args import wrap_arg
from numba.core.errors import NumbaPerformanceWarning
from .descriptor import cuda_target


def _nvvm_options_type(x):
    if x is None:
        return None

    else:
        assert isinstance(x, dict)
        return x


class CUDAFlags(Flags):
    nvvm_options = Option(
        type=_nvvm_options_type,
        default=None,
        doc="NVVM options",
    )


@register_pass(mutates_CFG=True, analysis_only=False)
class CUDABackend(LoweringPass):

    _name = "cuda_backend"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Packages lowering output in a compile result
        """
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)

        state.cr = compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=lowered.call_helper,
            signature=signature,
            fndesc=lowered.fndesc,
        )
        return True


@register_pass(mutates_CFG=False, analysis_only=False)
class CreateLibrary(LoweringPass):
    """
    Create a CUDACodeLibrary for the NativeLowering pass to populate. The
    NativeLowering pass will create a code library if none exists, but we need
    to set it up with nvvm_options from the flags if they are present.
    """

    _name = "create_library"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        codegen = state.targetctx.codegen()
        name = state.func_id.func_qualname
        nvvm_options = state.flags.nvvm_options
        state.library = codegen.create_library(name, nvvm_options=nvvm_options)
        # Enable object caching upfront so that the library can be serialized.
        state.library.enable_object_caching()

        return True


class CUDACompiler(CompilerBase):
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('cuda')

        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]

    def define_cuda_lowering_pipeline(self, state):
        pm = PassManager('cuda_lowering')
        # legalise
        pm.add_pass(IRLegalization,
                    "ensure IR is legal prior to lowering")

        # lower
        pm.add_pass(CreateLibrary, "create library")
        pm.add_pass(NativeLowering, "native lowering")
        pm.add_pass(CUDABackend, "cuda backend")

        pm.finalize()
        return pm


@global_compiler_lock
def compile_cuda(pyfunc, return_type, args, debug=False, lineinfo=False,
                 inline=False, fastmath=False, nvvm_options=None):
    from .descriptor import cuda_target
    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context

    flags = CUDAFlags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    if debug or lineinfo:
        # Note both debug and lineinfo turn on debug information in the
        # compiled code, but we keep them separate arguments in case we
        # later want to overload some other behavior on the debug flag.
        # In particular, -opt=3 is not supported with -g.
        flags.debuginfo = True
    if inline:
        flags.forceinline = True
    if fastmath:
        flags.fastmath = True
    if nvvm_options:
        flags.nvvm_options = nvvm_options

    # Run compilation pipeline
    cres = compiler.compile_extra(typingctx=typingctx,
                                  targetctx=targetctx,
                                  func=pyfunc,
                                  args=args,
                                  return_type=return_type,
                                  flags=flags,
                                  locals={},
                                  pipeline_class=CUDACompiler)

    library = cres.library
    library.finalize()

    return cres


def compile_kernel(pyfunc, args, link, debug=False, lineinfo=False,
                   inline=False, fastmath=False, extensions=[],
                   max_registers=None, opt=True):
    return _Kernel(pyfunc, args, link, debug=debug, lineinfo=lineinfo,
                   inline=inline, fastmath=fastmath, extensions=extensions,
                   max_registers=max_registers, opt=opt)


@global_compiler_lock
def compile_ptx(pyfunc, args, debug=False, lineinfo=False, device=False,
                fastmath=False, cc=None, opt=True):
    """Compile a Python function to PTX for a given set of argument types.

    :param pyfunc: The Python function to compile.
    :param args: A tuple of argument types to compile for.
    :param debug: Whether to include debug info in the generated PTX.
    :type debug: bool
    :param lineinfo: Whether to include a line mapping from the generated PTX
                     to the source code. Usually this is used with optimized
                     code (since debug mode would automatically include this),
                     so we want debug info in the LLVM but only the line
                     mapping in the final PTX.
    :type lineinfo: bool
    :param device: Whether to compile a device function. Defaults to ``False``,
                   to compile global kernel functions.
    :type device: bool
    :param fastmath: Whether to enable fast math flags (ftz=1, prec_sqrt=0,
                     prec_div=, and fma=1)
    :type fastmath: bool
    :param cc: Compute capability to compile for, as a tuple ``(MAJOR, MINOR)``.
               Defaults to ``(5, 2)``.
    :type cc: tuple
    :param opt: Enable optimizations. Defaults to ``True``.
    :type opt: bool
    :return: (ptx, resty): The PTX code and inferred return type
    :rtype: tuple
    """
    nvvm_options = {
        'debug': debug,
        'lineinfo': lineinfo,
        'fastmath': fastmath,
        'opt': 3 if opt else 0
    }

    cres = compile_cuda(pyfunc, None, args, debug=debug, lineinfo=lineinfo,
                        nvvm_options=nvvm_options)
    resty = cres.signature.return_type
    if device:
        lib = cres.library
    else:
        fname = cres.fndesc.llvm_func_name
        tgt = cres.target_context
        filename = cres.type_annotation.filename
        linenum = int(cres.type_annotation.linenum)
        lib, kernel = tgt.prepare_cuda_kernel(cres.library, fname,
                                              cres.signature.args, debug,
                                              nvvm_options, filename, linenum)

    cc = cc or config.CUDA_DEFAULT_PTX_CC
    ptx = lib.get_asm_str(cc=cc)
    return ptx, resty


def compile_ptx_for_current_device(pyfunc, args, debug=False, lineinfo=False,
                                   device=False, fastmath=False, opt=True):
    """Compile a Python function to PTX for a given set of argument types for
    the current device's compute capabilility. This calls :func:`compile_ptx`
    with an appropriate ``cc`` value for the current device."""
    cc = get_current_device().compute_capability
    return compile_ptx(pyfunc, args, debug=debug, lineinfo=lineinfo,
                       device=device, fastmath=fastmath, cc=cc, opt=True)


class DeviceDispatcher(serialize.ReduceMixin):
    """Unmaterialized device function
    """
    def __init__(self, pyfunc, debug, inline, opt):
        self.py_func = pyfunc
        self.debug = debug
        self.inline = inline
        self.opt = opt
        self.overloads = {}
        name = getattr(pyfunc, '__name__', 'unknown')
        self.__name__ = f"{name} <CUDA device function>".format(name)

    def _reduce_states(self):
        return dict(py_func=self.py_func, debug=self.debug, inline=self.inline)

    @classmethod
    def _rebuild(cls, py_func, debug, inline):
        return compile_device_dispatcher(py_func, debug=debug, inline=inline)

    def get_call_template(self, args, kws):
        # Copied and simplified from _DispatcherBase.get_call_template.
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows to resolve the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        # Ensure an overload is available
        self.compile(tuple(args))

        # Create function type for typing
        func_name = self.py_func.__name__
        name = "CallTemplate({0})".format(func_name)

        # The `key` isn't really used except for diagnosis here,
        # so avoid keeping a reference to `cfunc`.
        call_template = typing.make_concrete_template(
            name, key=func_name, signatures=self.nopython_signatures)
        pysig = utils.pysignature(self.py_func)

        return call_template, pysig, args, kws

    @property
    def nopython_signatures(self):
        # All overloads are for nopython mode, because there is only
        # nopython mode in CUDA
        return [info.signature for info in self.overloads.values()]

    def get_overload(self, sig):
        # NOTE: This dispatcher seems to be used as the key for the dict of
        # implementations elsewhere in Numba, so we return this dispatcher
        # instead of a compiled entry point as in
        # _DispatcherBase.get_overload().
        return self

    def compile(self, args):
        """Compile the function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
        if args not in self.overloads:
            nvvm_options = {
                'debug': self.debug,
                'opt': 3 if self.opt else 0
            }

            cres = compile_cuda(self.py_func, None, args, debug=self.debug,
                                inline=self.inline, nvvm_options=nvvm_options)
            first_definition = not self.overloads
            self.overloads[args] = cres
            libs = [cres.library]

            if first_definition:
                # First definition
                cres.target_context.insert_user_function(self, cres.fndesc,
                                                         libs)
            else:
                cres.target_context.add_user_function(self, cres.fndesc, libs)

        else:
            cres = self.overloads[args]

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
        modules = self.compile(args).library.modules
        return "\n\n".join([str(mod) for mod in modules])

    def inspect_ptx(self, args, nvvm_options={}):
        """Returns the PTX compiled for *args* for the currently active GPU

        Parameters
        ----------
        args: tuple[Type]
            Argument types.

        Returns
        -------
        ptx : bytes
        """
        msg = ('inspect_ptx for device functions is deprecated. Use '
               'compile_ptx instead.')
        warn(msg, category=NumbaDeprecationWarning)

        if nvvm_options:
            msg = ('nvvm_options are ignored. Use compile_ptx if you want to '
                   'set NVVM options.')
            warn(msg, category=NumbaDeprecationWarning)
        return self.compile(args).library.get_asm_str().encode()


def compile_device_dispatcher(pyfunc, debug=False, inline=False, opt=True):
    """Create a DeviceDispatcher and register it to the CUDA typing context.
    """
    from .descriptor import cuda_target

    dispatcher = DeviceDispatcher(pyfunc, debug=debug, inline=inline, opt=opt)

    class device_function_template(AbstractTemplate):
        key = dispatcher

        def generic(self, args, kws):
            assert not kws
            return dispatcher.compile(args).signature

        def get_template_info(cls):
            basepath = os.path.dirname(os.path.dirname(numba.__file__))
            code, firstlineno = inspect.getsourcelines(pyfunc)
            path = inspect.getsourcefile(pyfunc)
            sig = str(utils.pysignature(pyfunc))
            info = {
                'kind': "overload",
                'name': getattr(cls.key, '__name__', "unknown"),
                'sig': sig,
                'filename': utils.safe_relpath(path, start=basepath),
                'lines': (firstlineno, firstlineno + len(code) - 1),
                'docstring': pyfunc.__doc__
            }
            return info

    typingctx = cuda_target.typing_context
    typingctx.insert_user_function(dispatcher, device_function_template)
    return dispatcher


def compile_device(pyfunc, return_type, args, inline=True, debug=False,
                   lineinfo=False):
    return DeviceFunction(pyfunc, return_type, args, inline=True, debug=False,
                          lineinfo=False)


def declare_device_function(name, restype, argtypes):
    from .descriptor import cuda_target
    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context
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


class DeviceFunction(serialize.ReduceMixin):

    def __init__(self, pyfunc, return_type, args, inline, debug, lineinfo):
        self.py_func = pyfunc
        self.return_type = return_type
        self.args = args
        self.inline = True
        self.debug = False
        self.lineinfo = False
        cres = compile_cuda(self.py_func, self.return_type, self.args,
                            debug=self.debug, inline=self.inline,
                            lineinfo=self.lineinfo)
        self.cres = cres

        class device_function_template(ConcreteTemplate):
            key = self
            cases = [cres.signature]

        cres.typing_context.insert_user_function(
            self, device_function_template)
        cres.target_context.insert_user_function(self, cres.fndesc,
                                                 [cres.library])

    def _reduce_states(self):
        return dict(py_func=self.py_func, return_type=self.return_type,
                    args=self.args, inline=self.inline, debug=self.debug,
                    lineinfo=self.lineinfo)

    @classmethod
    def _rebuild(cls, py_func, return_type, args, inline, debug, lineinfo):
        return cls(py_func, return_type, args, inline, debug, lineinfo)

    def __repr__(self):
        fmt = "<DeviceFunction py_func={0} signature={1}>"
        return fmt.format(self.py_func, self.cres.signature)


class ExternFunction(object):
    def __init__(self, name, sig):
        self.name = name
        self.sig = sig


class ForAll(object):
    def __init__(self, kernel, ntasks, tpb, stream, sharedmem):
        if ntasks < 0:
            raise ValueError("Can't create ForAll with negative task count: %s"
                             % ntasks)
        self.kernel = kernel
        self.ntasks = ntasks
        self.thread_per_block = tpb
        self.stream = stream
        self.sharedmem = sharedmem

    def __call__(self, *args):
        if self.ntasks == 0:
            return

        if self.kernel.specialized:
            kernel = self.kernel
        else:
            kernel = self.kernel.specialize(*args)
        blockdim = self._compute_thread_per_block(kernel)
        griddim = (self.ntasks + blockdim - 1) // blockdim

        return kernel[griddim, blockdim, self.stream, self.sharedmem](*args)

    def _compute_thread_per_block(self, kernel):
        tpb = self.thread_per_block
        # Prefer user-specified config
        if tpb != 0:
            return tpb
        # Else, ask the driver to give a good config
        else:
            ctx = get_context()
            # Kernel is specialized, so there's only one definition - get it so
            # we can get the cufunc from the code library
            defn = next(iter(kernel.overloads.values()))
            kwargs = dict(
                func=defn._codelibrary.get_cufunc(),
                b2d_func=0,     # dynamic-shared memory is constant to blksz
                memsize=self.sharedmem,
                blocksizelimit=1024,
            )
            _, tpb = ctx.get_max_potential_block_size(**kwargs)
            return tpb


class _Kernel(serialize.ReduceMixin):
    '''
    CUDA Kernel specialized for a given set of argument types. When called, this
    object launches the kernel on the device.
    '''

    @global_compiler_lock
    def __init__(self, py_func, argtypes, link=None, debug=False,
                 lineinfo=False, inline=False, fastmath=False, extensions=None,
                 max_registers=None, opt=True):
        super().__init__()

        self.py_func = py_func
        self.argtypes = argtypes
        self.debug = debug
        self.lineinfo = lineinfo
        self.extensions = extensions or []

        nvvm_options = {
            'debug': self.debug,
            'lineinfo': self.lineinfo,
            'fastmath': fastmath,
            'opt': 3 if opt else 0
        }

        cres = compile_cuda(self.py_func, types.void, self.argtypes,
                            debug=self.debug,
                            lineinfo=self.lineinfo,
                            inline=inline,
                            fastmath=fastmath,
                            nvvm_options=nvvm_options)
        fname = cres.fndesc.llvm_func_name
        args = cres.signature.args

        tgt_ctx = cres.target_context
        code = self.py_func.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno
        lib, kernel = tgt_ctx.prepare_cuda_kernel(cres.library, fname, args,
                                                  debug, nvvm_options,
                                                  filename, linenum,
                                                  max_registers)

        if not link:
            link = []

        # A kernel needs cooperative launch if grid_sync is being used.
        self.cooperative = 'cudaCGGetIntrinsicHandle' in lib.get_asm_str()
        # We need to link against cudadevrt if grid sync is being used.
        if self.cooperative:
            link.append(get_cudalib('cudadevrt', static=True))

        for filepath in link:
            lib.add_linking_file(filepath)

        # populate members
        self.entry_name = kernel.name
        self.signature = cres.signature
        self._type_annotation = cres.type_annotation
        self._codelibrary = lib
        self.call_helper = cres.call_helper

    @property
    def argument_types(self):
        return tuple(self.signature.args)

    @classmethod
    def _rebuild(cls, cooperative, name, argtypes, codelibrary, link, debug,
                 lineinfo, call_helper, extensions):
        """
        Rebuild an instance.
        """
        instance = cls.__new__(cls)
        # invoke parent constructor
        super(cls, instance).__init__()
        # populate members
        instance.cooperative = cooperative
        instance.entry_name = name
        instance.argument_types = tuple(argtypes)
        instance._type_annotation = None
        instance._codelibrary = codelibrary
        instance.debug = debug
        instance.lineinfo = lineinfo
        instance.call_helper = call_helper
        instance.extensions = extensions
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are serialized in PTX form.
        Type annotation are discarded.
        Thread, block and shared memory configuration are serialized.
        Stream information is discarded.
        """
        return dict(cooperative=self.cooperative, name=self.entry_name,
                    argtypes=self.argtypes, codelibrary=self.codelibrary,
                    debug=self.debug, lineinfo=self.lineinfo,
                    call_helper=self.call_helper, extensions=self.extensions)

    def bind(self):
        """
        Force binding to current CUDA context
        """
        self._codelibrary.get_cufunc()

    @property
    def ptx(self):
        '''
        PTX code for this kernel.
        '''
        return self._codelibrary.get_asm_str()

    @property
    def device(self):
        """
        Get current active context
        """
        return get_current_device()

    @property
    def regs_per_thread(self):
        '''
        The number of registers used by each thread for this kernel.
        '''
        return self._codelibrary.get_cufunc().attrs.regs

    def inspect_llvm(self):
        '''
        Returns the LLVM IR for this kernel.
        '''
        return self._codelibrary.get_llvm_str()

    def inspect_asm(self, cc):
        '''
        Returns the PTX code for this kernel.
        '''
        return self._codelibrary.get_asm_str(cc=cc)

    def inspect_sass(self):
        '''
        Returns the SASS code for this kernel.

        Requires nvdisasm to be available on the PATH.
        '''
        return self._codelibrary.get_sass()

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

    def max_cooperative_grid_blocks(self, blockdim, dynsmemsize=0):
        '''
        Calculates the maximum number of blocks that can be launched for this
        kernel in a cooperative grid in the current context, for the given block
        and dynamic shared memory sizes.

        :param blockdim: Block dimensions, either as a scalar for a 1D block, or
                         a tuple for 2D or 3D blocks.
        :param dynsmemsize: Dynamic shared memory size in bytes.
        :return: The maximum number of blocks in the grid.
        '''
        ctx = get_context()
        cufunc = self._codelibrary.get_cufunc()

        if isinstance(blockdim, tuple):
            blockdim = functools.reduce(lambda x, y: x * y, blockdim)
        active_per_sm = ctx.get_active_blocks_per_multiprocessor(cufunc,
                                                                 blockdim,
                                                                 dynsmemsize)
        sm_count = ctx.device.MULTIPROCESSOR_COUNT
        return active_per_sm * sm_count

    def launch(self, args, griddim, blockdim, stream=0, sharedmem=0):
        # Prepare kernel
        cufunc = self._codelibrary.get_cufunc()

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

        stream_handle = stream and stream.handle or None

        # Invoke kernel
        driver.launch_kernel(cufunc.handle,
                             *griddim,
                             *blockdim,
                             sharedmem,
                             stream_handle,
                             kernelargs,
                             cooperative=self.cooperative)

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
                    filepath = os.path.abspath(filepath)
                    locinfo = 'In function %r, file %s, line %s, ' % (sym,
                                                                      filepath,
                                                                      lineno,)
                # Prefix the exception message with the thread position
                prefix = "%stid=%s ctaid=%s" % (locinfo, tid, ctaid)
                if exc_args:
                    exc_args = ("%s: %s" % (prefix, exc_args[0]),) + \
                        exc_args[1:]
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

        elif isinstance(ty, types.BaseTuple):
            assert len(ty) == len(val)
            for t, v in zip(ty, val):
                self._prepare_args(t, v, stream, retr, kernelargs)

        else:
            raise NotImplementedError(ty, val)


class _KernelConfiguration:
    def __init__(self, dispatcher, griddim, blockdim, stream, sharedmem):
        self.dispatcher = dispatcher
        self.griddim = griddim
        self.blockdim = blockdim
        self.stream = stream
        self.sharedmem = sharedmem

        if config.CUDA_LOW_OCCUPANCY_WARNINGS:
            ctx = get_context()
            smcount = ctx.device.MULTIPROCESSOR_COUNT
            grid_size = griddim[0] * griddim[1] * griddim[2]
            if grid_size < 2 * smcount:
                msg = ("Grid size ({grid}) < 2 * SM count ({sm}) "
                       "will likely result in GPU under utilization due "
                       "to low occupancy.")
                msg = msg.format(grid=grid_size, sm=2 * smcount)
                warn(NumbaPerformanceWarning(msg))

    def __call__(self, *args):
        return self.dispatcher.call(args, self.griddim, self.blockdim,
                                    self.stream, self.sharedmem)


class Dispatcher(_dispatcher.Dispatcher, serialize.ReduceMixin):
    '''
    CUDA Dispatcher object. When configured and called, the dispatcher will
    specialize itself for the given arguments (if no suitable specialized
    version already exists) & compute capability, and launch on the device
    associated with the current context.

    Dispatcher objects are not to be constructed by the user, but instead are
    created using the :func:`numba.cuda.jit` decorator.
    '''

    # Whether to fold named arguments and default values. Default values are
    # presently unsupported on CUDA, so we can leave this as False in all
    # cases.
    _fold_args = False

    targetdescr = cuda_target

    def __init__(self, py_func, sigs, targetoptions):
        self.py_func = py_func
        self.sigs = []
        self.link = targetoptions.pop('link', (),)
        self._can_compile = True

        # Specializations for given sets of argument types
        self.specializations = {}

        # A mapping of signatures to compile results
        self.overloads = collections.OrderedDict()

        self.targetoptions = targetoptions

        # defensive copy
        self.targetoptions['extensions'] = \
            list(self.targetoptions.get('extensions', []))

        self.typingctx = self.targetdescr.typing_context

        self._tm = default_type_manager

        pysig = utils.pysignature(py_func)
        arg_count = len(pysig.parameters)
        argnames = tuple(pysig.parameters)
        default_values = self.py_func.__defaults__ or ()
        defargs = tuple(OmittedArg(val) for val in default_values)
        can_fallback = False # CUDA cannot fallback to object mode

        try:
            lastarg = list(pysig.parameters.values())[-1]
        except IndexError:
            has_stararg = False
        else:
            has_stararg = lastarg.kind == lastarg.VAR_POSITIONAL

        exact_match_required = False

        _dispatcher.Dispatcher.__init__(self, self._tm.get_pointer(),
                                        arg_count, self._fold_args, argnames,
                                        defargs, can_fallback, has_stararg,
                                        exact_match_required)

        if sigs:
            if len(sigs) > 1:
                raise TypeError("Only one signature supported at present")
            self.compile(sigs[0])
            self._can_compile = False

    def configure(self, griddim, blockdim, stream=0, sharedmem=0):
        griddim, blockdim = normalize_kernel_dimensions(griddim, blockdim)
        return _KernelConfiguration(self, griddim, blockdim, stream, sharedmem)

    def __getitem__(self, args):
        if len(args) not in [2, 3, 4]:
            raise ValueError('must specify at least the griddim and blockdim')
        return self.configure(*args)

    def forall(self, ntasks, tpb=0, stream=0, sharedmem=0):
        """Returns a 1D-configured kernel for a given number of tasks.

        This assumes that:

        - the kernel maps the Global Thread ID ``cuda.grid(1)`` to tasks on a
          1-1 basis.
        - the kernel checks that the Global Thread ID is upper-bounded by
          ``ntasks``, and does nothing if it is not.

        :param ntasks: The number of tasks.
        :param tpb: The size of a block. An appropriate value is chosen if this
                    parameter is not supplied.
        :param stream: The stream on which the configured kernel will be
                       launched.
        :param sharedmem: The number of bytes of dynamic shared memory required
                          by the kernel.
        :return: A configured kernel, ready to launch on a set of arguments."""

        return ForAll(self, ntasks, tpb=tpb, stream=stream, sharedmem=sharedmem)

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

    def __call__(self, *args, **kwargs):
        # An attempt to launch an unconfigured kernel
        raise ValueError(missing_launch_config_msg)

    def call(self, args, griddim, blockdim, stream, sharedmem):
        '''
        Compile if necessary and invoke this kernel with *args*.
        '''
        if self.specialized:
            kernel = next(iter(self.overloads.values()))
        else:
            kernel = _dispatcher.Dispatcher._cuda_call(self, *args)

        kernel.launch(args, griddim, blockdim, stream, sharedmem)

    def _compile_for_args(self, *args, **kws):
        # Based on _DispatcherBase._compile_for_args.
        assert not kws
        argtypes = [self.typeof_pyval(a) for a in args]
        return self.compile(tuple(argtypes))

    def _search_new_conversions(self, *args, **kws):
        # Based on _DispatcherBase._search_new_conversions
        assert not kws
        args = [self.typeof_pyval(a) for a in args]
        found = False
        for sig in self.nopython_signatures:
            conv = self.typingctx.install_possible_conversions(args, sig.args)
            if conv:
                found = True
        return found

    def typeof_pyval(self, val):
        # Based on _DispatcherBase.typeof_pyval, but differs from it to support
        # the CUDA Array Interface.
        try:
            return typeof(val, Purpose.argument)
        except ValueError:
            if numba.cuda.is_cuda_array(val):
                # When typing, we don't need to synchronize on the array's
                # stream - this is done when the kernel is launched.
                return typeof(numba.cuda.as_cuda_array(val, sync=False),
                              Purpose.argument)
            else:
                raise

    @property
    def nopython_signatures(self):
        # Based on _DispatcherBase.nopython_signatures
        return [kernel.signature for kernel in self.overloads.values()]

    def specialize(self, *args):
        '''
        Create a new instance of this dispatcher specialized for the given
        *args*.
        '''
        cc = get_current_device().compute_capability
        argtypes = tuple(
            [self.typingctx.resolve_argument_type(a) for a in args])
        if self.specialized:
            raise RuntimeError('Dispatcher already specialized')

        specialization = self.specializations.get((cc, argtypes))
        if specialization:
            return specialization

        targetoptions = self.targetoptions
        targetoptions['link'] = self.link
        specialization = Dispatcher(self.py_func, [types.void(*argtypes)],
                                    targetoptions)
        self.specializations[cc, argtypes] = specialization
        return specialization

    def disable_compile(self, val=True):
        self._can_compile = not val

    @property
    def specialized(self):
        """
        True if the Dispatcher has been specialized.
        """
        return len(self.sigs) == 1 and not self._can_compile

    def get_regs_per_thread(self, signature=None):
        '''
        Returns the number of registers used by each thread in this kernel for
        the device in the current context.

        :param signature: The signature of the compiled kernel to get register
                          usage for. This may be omitted for a specialized
                          kernel.
        :return: The number of registers used by the compiled variant of the
                 kernel for the given signature and current device.
        '''
        if signature is not None:
            return self.overloads[signature.args].regs_per_thread
        if self.specialized:
            return next(iter(self.overloads.values())).regs_per_thread
        else:
            return {sig: overload.regs_per_thread
                    for sig, overload in self.overloads.items()}

    def compile(self, sig):
        '''
        Compile and bind to the current context a version of this kernel
        specialized for the given signature.
        '''
        argtypes, return_type = sigutils.normalize_signature(sig)
        assert return_type is None or return_type == types.none
        if self.specialized:
            return next(iter(self.overloads.values()))
        else:
            kernel = self.overloads.get(argtypes)
        if kernel is None:
            if not self._can_compile:
                raise RuntimeError("Compilation disabled")
            kernel = _Kernel(self.py_func, argtypes, link=self.link,
                             **self.targetoptions)
            # Inspired by _DispatcherBase.add_overload, but differs slightly
            # because we're inserting a _Kernel object instead of a compiled
            # function.
            c_sig = [a._code for a in argtypes]
            self._insert(c_sig, kernel, cuda=True)
            self.overloads[argtypes] = kernel

            kernel.bind()
            self.sigs.append(sig)
        return kernel

    def inspect_llvm(self, signature=None):
        '''
        Return the LLVM IR for this kernel.

        :param signature: A tuple of argument types.
        :return: The LLVM IR for the given signature, or a dict of LLVM IR
                 for all previously-encountered signatures.

        '''
        if signature is not None:
            return self.overloads[signature].inspect_llvm()
        else:
            return {sig: overload.inspect_llvm()
                    for sig, overload in self.overloads.items()}

    def inspect_asm(self, signature=None):
        '''
        Return this kernel's PTX assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The PTX code for the given signature, or a dict of PTX codes
                 for all previously-encountered signatures.
        '''
        cc = get_current_device().compute_capability
        if signature is not None:
            return self.overloads[signature].inspect_asm(cc)
        else:
            return {sig: overload.inspect_asm(cc)
                    for sig, overload in self.overloads.items()}

    def inspect_sass(self, signature=None):
        '''
        Return this kernel's SASS assembly code for for the device in the
        current context.

        :param signature: A tuple of argument types.
        :return: The SASS code for the given signature, or a dict of SASS codes
                 for all previously-encountered signatures.

        SASS for the device in the current context is returned.

        Requires nvdisasm to be available on the PATH.
        '''
        if signature is not None:
            return self.overloads[signature].inspect_sass()
        else:
            return {sig: defn.inspect_sass()
                    for sig, defn in self.overloads.items()}

    def inspect_types(self, file=None):
        '''
        Produce a dump of the Python source of this function annotated with the
        corresponding Numba IR and type information. The dump is written to
        *file*, or *sys.stdout* if *file* is *None*.
        '''
        if file is None:
            file = sys.stdout

        for _, defn in self.overloads.items():
            defn.inspect_types(file=file)

    @property
    def ptx(self):
        return {sig: overload.ptx for sig, overload in self.overloads.items()}

    def bind(self):
        for defn in self.overloads.values():
            defn.bind()

    @classmethod
    def _rebuild(cls, py_func, sigs, targetoptions):
        """
        Rebuild an instance.
        """
        instance = cls(py_func, sigs, targetoptions)
        return instance

    def _reduce_states(self):
        """
        Reduce the instance for serialization.
        Compiled definitions are discarded.
        """
        return dict(py_func=self.py_func, sigs=self.sigs,
                    targetoptions=self.targetoptions)
