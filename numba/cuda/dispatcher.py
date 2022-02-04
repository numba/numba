import collections
import inspect
import numpy as np
import os
import sys

from numba.core import config, serialize, sigutils, types, typing, utils
from numba.core.dispatcher import CompilingCounter, OmittedArg
from numba.core.errors import NumbaPerformanceWarning
from numba.core.typeconv.rules import default_type_manager
from numba.core.typing.templates import AbstractTemplate
from numba.core.typing.typeof import Purpose, typeof

from numba.cuda.api import get_current_device
from numba.cuda.compiler import compile_cuda, ForAll, _Kernel
from numba.cuda.cudadrv.devices import get_context
from numba.cuda.descriptor import cuda_target
from numba.cuda.errors import (missing_launch_config_msg,
                               normalize_kernel_dimensions)
from numba.cuda import types as cuda_types

from numba import cuda
from numba import _dispatcher
from numba.cuda.cudadrv import devicearray
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GenerializedUFunc,
                                        GUFuncCallSteps)

from warnings import warn


class CUDAUFuncDispatcher(object):
    """
    Invoke the CUDA ufunc specialization for the given inputs.
    """

    def __init__(self, types_to_retty_kernels):
        self.functions = types_to_retty_kernels
        self._maxblocksize = 0  # ignored

    @property
    def max_blocksize(self):
        return self._maxblocksize

    @max_blocksize.setter
    def max_blocksize(self, blksz):
        self._max_blocksize = blksz

    def __call__(self, *args, **kws):
        """
        *args: numpy arrays or DeviceArrayBase (created by cuda.to_device).
               Cannot mix the two types in one call.

        **kws:
            stream -- cuda stream; when defined, asynchronous mode is used.
            out    -- output array. Can be a numpy array or DeviceArrayBase
                      depending on the input arguments.  Type must match
                      the input arguments.
        """
        return CUDAUFuncMechanism.call(self.functions, args, kws)

    def reduce(self, arg, stream=0):
        assert len(list(self.functions.keys())[0]) == 2, "must be a binary " \
                                                         "ufunc"
        assert arg.ndim == 1, "must use 1d array"

        n = arg.shape[0]
        gpu_mems = []

        if n == 0:
            raise TypeError("Reduction on an empty array.")
        elif n == 1:  # nothing to do
            return arg[0]

        # always use a stream
        stream = stream or cuda.stream()
        with stream.auto_synchronize():
            # transfer memory to device if necessary
            if devicearray.is_cuda_ndarray(arg):
                mem = arg
            else:
                mem = cuda.to_device(arg, stream)
                # do reduction
            out = self.__reduce(mem, gpu_mems, stream)
            # use a small buffer to store the result element
            buf = np.array((1,), dtype=arg.dtype)
            out.copy_to_host(buf, stream=stream)

        return buf[0]

    def __reduce(self, mem, gpu_mems, stream):
        n = mem.shape[0]
        if n % 2 != 0:  # odd?
            fatcut, thincut = mem.split(n - 1)
            # prevent freeing during async mode
            gpu_mems.append(fatcut)
            gpu_mems.append(thincut)
            # execute the kernel
            out = self.__reduce(fatcut, gpu_mems, stream)
            gpu_mems.append(out)
            return self(out, thincut, out=out, stream=stream)
        else:  # even?
            left, right = mem.split(n // 2)
            # prevent freeing during async mode
            gpu_mems.append(left)
            gpu_mems.append(right)
            # execute the kernel
            self(left, right, out=left, stream=stream)
            if n // 2 > 1:
                return self.__reduce(left, gpu_mems, stream)
            else:
                return left


class _CUDAGUFuncCallSteps(GUFuncCallSteps):
    __slots__ = [
        '_stream',
    ]

    def is_device_array(self, obj):
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        # We don't want to call as_cuda_array on objects that are already Numba
        # device arrays, because this results in exporting the array as a
        # Producer then importing it as a Consumer, which causes a
        # synchronization on the array's stream (if it has one) by default.
        # When we have a Numba device array, we can simply return it.
        if devicearray.is_cuda_ndarray(obj):
            return obj
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary):
        return cuda.to_device(hostary, stream=self._stream)

    def to_host(self, devary, hostary):
        out = devary.copy_to_host(hostary, stream=self._stream)
        return out

    def device_array(self, shape, dtype):
        return cuda.device_array(shape=shape, dtype=dtype, stream=self._stream)

    def prepare_inputs(self):
        self._stream = self.kwargs.get('stream', 0)

    def launch_kernel(self, kernel, nelem, args):
        kernel.forall(nelem, stream=self._stream)(*args)


class CUDAGenerializedUFunc(GenerializedUFunc):
    @property
    def _call_steps(self):
        return _CUDAGUFuncCallSteps

    def _broadcast_scalar_input(self, ary, shape):
        return devicearray.DeviceNDArray(shape=shape,
                                         strides=(0,),
                                         dtype=ary.dtype,
                                         gpu_data=ary.gpu_data)

    def _broadcast_add_axis(self, ary, newshape):
        newax = len(newshape) - len(ary.shape)
        # Add 0 strides for missing dimension
        newstrides = (0,) * newax + ary.strides
        return devicearray.DeviceNDArray(shape=newshape,
                                         strides=newstrides,
                                         dtype=ary.dtype,
                                         gpu_data=ary.gpu_data)


class CUDAUFuncMechanism(UFuncMechanism):
    """
    Provide CUDA specialization
    """
    DEFAULT_STREAM = 0

    def launch(self, func, count, stream, args):
        func.forall(count, stream=stream)(*args)

    def is_device_array(self, obj):
        return cuda.is_cuda_array(obj)

    def as_device_array(self, obj):
        # We don't want to call as_cuda_array on objects that are already Numba
        # device arrays, because this results in exporting the array as a
        # Producer then importing it as a Consumer, which causes a
        # synchronization on the array's stream (if it has one) by default.
        # When we have a Numba device array, we can simply return it.
        if devicearray.is_cuda_ndarray(obj):
            return obj
        return cuda.as_cuda_array(obj)

    def to_device(self, hostary, stream):
        return cuda.to_device(hostary, stream=stream)

    def to_host(self, devary, stream):
        return devary.copy_to_host(stream=stream)

    def device_array(self, shape, dtype, stream):
        return cuda.device_array(shape=shape, dtype=dtype, stream=stream)

    def broadcast_device(self, ary, shape):
        ax_differs = [ax for ax in range(len(shape))
                      if ax >= ary.ndim
                      or ary.shape[ax] != shape[ax]]

        missingdim = len(shape) - len(ary.shape)
        strides = [0] * missingdim + list(ary.strides)

        for ax in ax_differs:
            strides[ax] = 0

        return devicearray.DeviceNDArray(shape=shape,
                                         strides=strides,
                                         dtype=ary.dtype,
                                         gpu_data=ary.gpu_data)


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
        self._type = self._numba_type_

        # The compiling counter is only used when compiling device functions as
        # it is used to detect recursion - recursion is not possible when
        # compiling a kernel.
        self._compiling_counter = CompilingCounter()

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
            if targetoptions.get('device'):
                argtypes, restype = sigutils.normalize_signature(sigs[0])
                self.compile_device(argtypes)
            else:
                self.compile(sigs[0])

            self._can_compile = False

        if targetoptions.get('device'):
            self._register_device_function()

    def _register_device_function(self):
        dispatcher = self
        pyfunc = self.py_func

        class device_function_template(AbstractTemplate):
            key = dispatcher

            def generic(self, args, kws):
                assert not kws
                return dispatcher.compile(args).signature

            def get_template_info(cls):
                basepath = os.path.dirname(
                    os.path.dirname(os.path.dirname(cuda.__file__)))
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

        from .descriptor import cuda_target
        typingctx = cuda_target.typing_context
        typingctx.insert_user_function(dispatcher, device_function_template)

    @property
    def _numba_type_(self):
        return cuda_types.CUDADispatcher(self)

    @property
    def is_compiling(self):
        """
        Whether a specialization is currently being compiled.
        """
        return self._compiling_counter

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
        return self.targetoptions.get('extensions')

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
            if cuda.is_cuda_array(val):
                # When typing, we don't need to synchronize on the array's
                # stream - this is done when the kernel is launched.
                return typeof(cuda.as_cuda_array(val, sync=False),
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

    def get_call_template(self, args, kws):
        # Copied and simplified from _DispatcherBase.get_call_template.
        """
        Get a typing.ConcreteTemplate for this dispatcher and the given
        *args* and *kws* types.  This allows resolution of the return type.

        A (template, pysig, args, kws) tuple is returned.
        """
        with self._compiling_counter:
            # Ensure an exactly-matching overload is available if we can
            # compile. We proceed with the typing even if we can't compile
            # because we may be able to force a cast on the caller side.
            if self._can_compile:
                self.compile_device(tuple(args))

            # Create function type for typing
            func_name = self.py_func.__name__
            name = "CallTemplate({0})".format(func_name)

            call_template = typing.make_concrete_template(
                name, key=func_name, signatures=self.nopython_signatures)
            pysig = utils.pysignature(self.py_func)

            return call_template, pysig, args, kws

    def get_overload(self, sig):
        # We give the id of the overload (a CompileResult) because this is used
        # as a key into a dict of overloads, and this is the only small and
        # unique property of a CompileResult on CUDA (c.f. the CPU target,
        # which uses its entry_point, which is a pointer value).
        args, return_type = sigutils.normalize_signature(sig)
        return id(self.overloads[args])

    def compile_device(self, args):
        """Compile the device function for the given argument types.

        Each signature is compiled once by caching the compiled function inside
        this object.

        Returns the `CompileResult`.
        """
        if args not in self.overloads:

            debug = self.targetoptions.get('debug')
            inline = self.targetoptions.get('inline')

            nvvm_options = {
                'debug': debug,
                'opt': 3 if self.targetoptions.get('opt') else 0
            }

            cres = compile_cuda(self.py_func, None, args, debug=debug,
                                inline=inline, nvvm_options=nvvm_options)
            self.overloads[args] = cres

            # The inserted function uses the id of the CompileResult as a key,
            # consistent with get_overload() above.
            cres.target_context.insert_user_function(id(cres), cres.fndesc,
                                                     [cres.library])
        else:
            cres = self.overloads[args]

        return cres

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
        device = self.targetoptions.get('device')
        if signature is not None:
            if device:
                return self.overloads[signature].library.get_llvm_str()
            else:
                return self.overloads[signature].inspect_llvm()
        else:
            if device:
                return {sig: overload.library.get_llvm_str()
                        for sig, overload in self.overloads.items()}
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
        device = self.targetoptions.get('device')
        if signature is not None:
            if device:
                return self.overloads[signature].library.get_asm_str(cc)
            else:
                return self.overloads[signature].inspect_asm(cc)
        else:
            if device:
                return {sig: overload.library.get_asm_str(cc)
                        for sig, overload in self.overloads.items()}
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
        if self.targetoptions.get('device'):
            raise RuntimeError('Cannot inspect SASS of a device function')

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
