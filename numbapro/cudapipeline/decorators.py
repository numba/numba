import numba
from numba import numbawrapper, autojit as _numba_autojit
from .environment import CudaEnvironment

def cuda_jit(restype=None, argtypes=None, nopython=False,
             _llvm_module=None, env_name=None, env=None,
             device=False,  inline=False, **kwargs):
    # error handling
    if restype and restype != numba.void:
        raise TypeError("CUDA kernel must have void return type.")
    assert _llvm_module is None
    # real work
    nopython = True       # override nopython flag
    env_name = env_name or 'numbapro.cuda'
    env = env or CudaEnvironment.get_environment(env_name)
    def _jit_decorator(func):
        envsp = env.specializations
        envsp.register(func)
        result = envsp.compile_function(func, argtypes, restype=restype,
                                        nopython=nopython, ctypes=False,
                                        compile_only=True, **kwargs)
        sig, lfunc, pycallable = result
        assert pycallable is None

        if device:
            env.add_device_function(func, lfunc)
            wrappercls = CudaDeviceFunction
        else:
            assert not inline
            wrappercls = CudaNumbaFunction

        return wrappercls(env, func, signature=sig, lfunc=lfunc)

    return _jit_decorator


def cuda_autojit(*args, **kws):
    env = CudaEnvironment.get_environment('numbapro.cuda')
    kws.setdefault('env', env)
    kws['nopython'] = True
    kws['target'] = 'gpu'
    return _numba_autojit(*args, **kws)

#
# Wrapper function
#

class CudaDeviceFunction(numbawrapper.NumbaWrapper):
    def __init__(self, env, py_func, signature, lfunc):
        super(CudaDeviceFunction, self).__init__(py_func)
        self.signature = signature
        self.lfunc = lfunc
        # print 'translating...'
        self.module = lfunc.module
        self.env = env

    def __call__(self, *args, **kws):
        raise TypeError("")


class CudaBaseFunction(numbawrapper.NumbaWrapper):

    _griddim = 1, 1, 1      # default grid dimension
    _blockdim = 1, 1, 1     # default block dimension
    _stream = 0
    
    def configure(self, griddim, blockdim, stream=0):
        '''Returns a new instance that is configured with the
            specified kernel grid dimension and block dimension.

            griddim, blockdim -- Triples of at most 3 integers. Missing dimensions
            are automatically filled with `1`.

            '''
        import copy
        inst = copy.copy(self) # clone the object

        inst._griddim = griddim
        inst._blockdim = blockdim

        while len(inst._griddim) < 3:
            inst._griddim += (1,)

        while len(inst._blockdim) < 3:
            inst._blockdim += (1,)

        inst._stream = stream

        return inst

    def __getitem__(self, args):
        '''Shorthand for self.configure()
            '''
        return self.configure(*args)



class CudaNumbaFunction(CudaBaseFunction):
    def __init__(self, env, py_func, signature, lfunc):
        super(CudaNumbaFunction, self).__init__(py_func)
        self.signature = signature
        self.lfunc = lfunc
        # print 'translating...'
        self.module = lfunc.module
        # self.extra = extra # unused
        self.env = env
        func_name = lfunc.name

        # FIXME: this function is called multiple times on the same lfunc.
        #        As a result, it has a long list of nvvm.annotation of the
        #        same data.

        self._ptxasm = self.env.ptxutils.generate_ptx(lfunc.module, [lfunc])

        from numbapro import _cudadispatch
        self.dispatcher = _cudadispatch.CudaNumbaFuncDispatcher(self._ptxasm,
                                                                func_name,
                                                                lfunc.type)

    def __call__(self, *args):
        '''Call the CUDA kernel.

            This call is synchronous to the host.
            In another words, this function will return only upon the completion
            of the CUDA kernel.
            '''
        return self.dispatcher(args, self._griddim, self._blockdim,
                               stream=self._stream)

    #    @property
    #    def compute_capability(self):
    #        '''The compute_capability of the PTX is generated for.
    #        '''
    #        return self._cc

    #    @property
    #    def target_machine(self):
    #        '''The LLVM target mcahine backend used to generate the PTX.
    #        '''
    #        return self._arch
    #
    @property
    def ptx(self):
        '''Returns the PTX assembly for this function.
            '''
        return self._ptxasm

    @property
    def device(self):
        return self.dispatcher.device


class CudaAutoJitNumbaFunction(CudaBaseFunction):

    def __init__(self, py_func, compiling_decorator, funccache):
        super(CudaAutoJitNumbaFunction, self).__init__(py_func)
        self.compiling_decorator = compiling_decorator

    def __call__(self, *args, **kwargs):
        if len(kwargs):
             raise error.NumbaError("Cannot handle keyword arguments yet")
        numba_wrapper = self.compiling_decorator(args, kwargs)
        return numba_wrapper[self._griddim, self._blockdim](*args)


