import ctypes
import functools
import os
import sys

import numpy as np

from numba.core.typing.templates import ConcreteTemplate
from numba.core import (types, typing, funcdesc, serialize, config,
                        compiler)
from numba.core.compiler import (CompilerBase, DefaultPassBuilder,
                                 compile_result, Flags, Option)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass, AnalysisPass,
                                           PassManager, register_pass)
from numba.core.errors import NumbaInvalidConfigWarning, TypingError
from numba.core.typed_passes import (IRLegalization, NativeLowering,
                                     AnnotateTypes)
from warnings import warn
from .cudadrv.devices import get_context
from .cudadrv.libs import get_cudalib
from .cudadrv import driver
from .api import get_current_device
from .args import wrap_arg


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


@register_pass(mutates_CFG=False, analysis_only=True)
class CUDALegalization(AnalysisPass):

    _name = "cuda_legalization"

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        # Early return if NVVM 7
        from numba.cuda.cudadrv.nvvm import NVVM
        if NVVM().is_nvvm70:
            return False
        # NVVM < 7, need to check for charseq
        typmap = state.typemap

        def check_dtype(dtype):
            if isinstance(dtype, (types.UnicodeCharSeq, types.CharSeq)):
                msg = (f"{k} is a char sequence type. This type is not "
                       "supported with CUDA toolkit versions < 11.2. To "
                       "use this type, you need to update your CUDA "
                       "toolkit - try 'conda install cudatoolkit=11' if "
                       "you are using conda to manage your environment.")
                raise TypingError(msg)
            elif isinstance(dtype, types.Record):
                for subdtype in dtype.fields.items():
                    # subdtype is a (name, _RecordField) pair
                    check_dtype(subdtype[1].type)

        for k, v in typmap.items():
            if isinstance(v, types.Array):
                check_dtype(v.dtype)
        return False


class CUDACompiler(CompilerBase):
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('cuda')

        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)
        pm.add_pass(CUDALegalization, "CUDA legalization")

        lowering_passes = self.define_cuda_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]

    def define_cuda_lowering_pipeline(self, state):
        pm = PassManager('cuda_lowering')
        # legalise
        pm.add_pass(IRLegalization,
                    "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")

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
    if debug and opt:
        msg = ("debug=True with opt=True (the default) "
               "is not supported by CUDA. This may result in a crash"
               " - set debug=False or opt=False.")
        warn(NumbaInvalidConfigWarning(msg))

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
        tgt = cres.target_context
        filename = cres.type_annotation.filename
        linenum = int(cres.type_annotation.linenum)
        lib, kernel = tgt.prepare_cuda_kernel(cres.library, cres.fndesc, debug,
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
                 max_registers=None, opt=True, device=False):

        if device:
            raise RuntimeError('Cannot compile a device function as a kernel')

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
        tgt_ctx = cres.target_context
        code = self.py_func.__code__
        filename = code.co_filename
        linenum = code.co_firstlineno
        lib, kernel = tgt_ctx.prepare_cuda_kernel(cres.library, cres.fndesc,
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

        if driver.USE_NV_BINDING:
            zero_stream = driver.binding.CUstream(0)
        else:
            zero_stream = None

        stream_handle = stream and stream.handle or zero_stream

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

            ptr = driver.device_pointer(devary)

            if driver.USE_NV_BINDING:
                ptr = int(ptr)

            data = ctypes.c_void_p(ptr)

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

        elif ty == types.float16:
            cval = ctypes.c_uint16(np.float16(val).view(np.uint16))
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
            ptr = devrec.device_ctypes_pointer
            if driver.USE_NV_BINDING:
                ptr = ctypes.c_void_p(int(ptr))
            kernelargs.append(ptr)

        elif isinstance(ty, types.BaseTuple):
            assert len(ty) == len(val)
            for t, v in zip(ty, val):
                self._prepare_args(t, v, stream, retr, kernelargs)

        else:
            raise NotImplementedError(ty, val)
