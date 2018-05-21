from gumath import unsafe_add_kernel
from ndtypes import ndt

from .. import jit
from .llvm import build_kernel_wrapper
from .ndtypes import Function, DType

class GuFunc:
    i = 0

    def __init__(self, fn, **kwargs):
        self.dispatcher =  jit(**kwargs)(fn)
        self.name = f'numba.{GuFunc.i}'
        GuFunc.i += 1
        self.already_added = set()
        self.func = None

    def add(self, f: Function):
        if f in self.already_added:
            return
        
        # numba gives us back the function, but we want the compile result
        # so we search for it
        entry_point = self.dispatcher.compile(f.as_numba)
        cres = [cres for cres in self.dispatcher.overloads.values() if cres.entry_point == entry_point][0]

        # replace the return value with the numba jitted one, if we can
        if f.returns_scalar:
            f_new = f.make_output_concrete(DType(str(cres.signature.return_type)))
        else:
            f_new = f
        func = unsafe_add_kernel(
            name=self.name,
            sig=f_new.as_ndt,
            ptr=build_kernel_wrapper(cres, f_new.dimensions),
            tag='Xnd'
        )
        if self.func is None:
            self.func = func
        self.already_added.add(f)
        return cres

    def __call__(self, *args):
        self.add(Function.zero_dim([DType(str(a.type.hidden_dtype)) for a in args]))
        return self.func(*args)
