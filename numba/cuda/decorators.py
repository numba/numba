from warnings import warn
from numba.core import types, config, sigutils
from numba.core.errors import DeprecationError, NumbaInvalidConfigWarning
from .compiler import declare_device_function, Dispatcher
from .simulator.kernel import FakeCUDAKernel


_msg_deprecated_signature_arg = ("Deprecated keyword argument `{0}`. "
                                 "Signatures should be passed as the first "
                                 "positional argument.")


def jit(func_or_sig=None, device=False, inline=False, link=[], debug=None,
        opt=True, **kws):
    """
    JIT compile a python function conforming to the CUDA Python specification.
    If a signature is supplied, then a function is returned that takes a
    function to compile.

    :param func_or_sig: A function to JIT compile, or a signature of a function
       to compile. If a function is supplied, then a
       :class:`numba.cuda.compiler.AutoJitCUDAKernel` is returned. If a
       signature is supplied, then a function is returned. The returned
       function accepts another function, which it will compile and then return
       a :class:`numba.cuda.compiler.AutoJitCUDAKernel`.

       .. note:: A kernel cannot have any return value.
    :param device: Indicates whether this is a device function.
    :type device: bool
    :param link: A list of files containing PTX source to link with the function
    :type link: list
    :param debug: If True, check for exceptions thrown when executing the
       kernel. Since this degrades performance, this should only be used for
       debugging purposes. If set to True, then ``opt`` should be set to False.
       Defaults to False.  (The default value can be overridden by setting
       environment variable ``NUMBA_CUDA_DEBUGINFO=1``.)
    :param fastmath: When True, enables fastmath optimizations as outlined in
       the :ref:`CUDA Fast Math documentation <cuda-fast-math>`.
    :param max_registers: Request that the kernel is limited to using at most
       this number of registers per thread. The limit may not be respected if
       the ABI requires a greater number of registers than that requested.
       Useful for increasing occupancy.
    :param opt: Whether to compile from LLVM IR to PTX with optimization
                enabled. When ``True``, ``-opt=3`` is passed to NVVM. When
                ``False``, ``-opt=0`` is passed to NVVM. Defaults to ``True``.
    :type opt: bool
    :param lineinfo: If True, generate a line mapping between source code and
       assembly code. This enables inspection of the source code in NVIDIA
       profiling tools and correlation with program counter sampling.
    :type lineinfo: bool
    """

    if link and config.ENABLE_CUDASIM:
        raise NotImplementedError('Cannot link PTX in the simulator')

    if kws.get('boundscheck'):
        raise NotImplementedError("bounds checking is not supported for CUDA")

    if kws.get('argtypes') is not None:
        msg = _msg_deprecated_signature_arg.format('argtypes')
        raise DeprecationError(msg)
    if kws.get('restype') is not None:
        msg = _msg_deprecated_signature_arg.format('restype')
        raise DeprecationError(msg)
    if kws.get('bind') is not None:
        msg = _msg_deprecated_signature_arg.format('bind')
        raise DeprecationError(msg)

    debug = config.CUDA_DEBUGINFO_DEFAULT if debug is None else debug
    fastmath = kws.get('fastmath', False)

    if debug and opt:
        msg = ("debug=True with opt=True (the default) "
               "is not supported by CUDA. This may result in a crash"
               " - set debug=False or opt=False.")
        warn(NumbaInvalidConfigWarning(msg))

    if device and kws.get('link'):
        raise ValueError("link keyword invalid for device function")

    if sigutils.is_signature(func_or_sig):
        if config.ENABLE_CUDASIM:
            def jitwrapper(func):
                return FakeCUDAKernel(func, device=device, fastmath=fastmath)
            return jitwrapper

        argtypes, restype = sigutils.normalize_signature(func_or_sig)

        if restype and not device and restype != types.void:
            raise TypeError("CUDA kernel must have void return type.")

        def _jit(func):
            targetoptions = kws.copy()
            targetoptions['debug'] = debug
            targetoptions['link'] = link
            targetoptions['opt'] = opt
            targetoptions['fastmath'] = fastmath
            targetoptions['device'] = device
            return Dispatcher(func, [func_or_sig], targetoptions=targetoptions)

        return _jit
    else:
        if func_or_sig is None:
            if config.ENABLE_CUDASIM:
                def autojitwrapper(func):
                    return FakeCUDAKernel(func, device=device,
                                          fastmath=fastmath)
            else:
                def autojitwrapper(func):
                    return jit(func, device=device, debug=debug, opt=opt,
                               link=link, **kws)

            return autojitwrapper
        # func_or_sig is a function
        else:
            if config.ENABLE_CUDASIM:
                return FakeCUDAKernel(func_or_sig, device=device,
                                      fastmath=fastmath)
            else:
                targetoptions = kws.copy()
                targetoptions['debug'] = debug
                targetoptions['opt'] = opt
                targetoptions['link'] = link
                targetoptions['fastmath'] = fastmath
                targetoptions['device'] = device
                sigs = None
                return Dispatcher(func_or_sig, sigs,
                                  targetoptions=targetoptions)


def declare_device(name, sig):
    argtypes, restype = sigutils.normalize_signature(sig)
    return declare_device_function(name, restype, argtypes)
