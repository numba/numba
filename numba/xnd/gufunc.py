from gumath import unsafe_add_kernel
from ndtypes import ndt

from .. import jit
from .llvm import build_kernel_wrapper
from .ndtypes import Function, DType

class GuFunc:
    """
    Enables creation of a gumath function, based on jitting a Python function.

    You pass in the function you want to JIT, then call `compile` for each
    type of kernel you want it to add to the function.

    It exposes the gumath function as the `func` attribute, which can then
    be called on XND types to execute it.
    """
    i = 0

    def __init__(self, fn):
        self.dispatcher =  jit(fn)

        # name must be unique
        self.name = f'numba.{GuFunc.i}'
        GuFunc.i += 1

        self.already_compiled = set()
        self.func = None

    def compile(self, f: Function):
        if f in self.already_compiled:
            return
        
        # numba gives us back the function, but we want the compile result
        # so we search for it
        entry_point = self.dispatcher.compile(f.as_numba)
        cres = [cres for cres in self.dispatcher.overloads.values() if cres.entry_point == entry_point][0]
        if cres.objectmode:
            raise NotImplementedError('Python/object mode not supported')

        # replace the return value with the numba jitted one, if we can
        if f.returns_scalar:
            f_new = f.make_output_concrete(DType(str(cres.signature.return_type)))
        else:
            f_new = f
        
        # gumath passes back the function after adding the kernel
        func = unsafe_add_kernel(
            name=self.name,
            sig=f_new.as_ndt,
            ptr=build_kernel_wrapper(cres, f_new.dimensions),
            tag='Xnd'
        )
        # we only save it if this is our first compilation and we don't have it already.
        # it's the same function every time we compile, since the name is the same.
        if self.func is None:
            self.func = func
        self.already_compiled.add(f)
        return cres

    def __call__(self, *args):
        '''
        Compile a kernel using the dtypes of the input arguments.
        '''
        self.compile(Function.zero_dim([DType(str(a.type.hidden_dtype)) for a in args]))
        return self.func(*args)
