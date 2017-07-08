from numba import types
from numba.typing.templates import infer_global, AbstractTemplate
from numba.typing import signature

def stencil():
    pass

@infer_global(stencil)
class Stencil(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(types.none, *args)
