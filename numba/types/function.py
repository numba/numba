
from .abstract import Type

class FunctionType(Type):
    """
    Represents a first-class function type.
    """
    mutable = True
    cconv = None

    def __init__(self, rtype, atypes):
        from numba.function import mangle
        self.rtype = rtype
        self.atypes = tuple(atypes)
        name = 'FT'+mangle(self)
        super(FunctionType, self).__init__(name)

    @property
    def key(self):
        return self.name

    def cast_python_value(self, value):
        raise NotImplementedError(f'cast_python_value({value})')

    def get_call_type(self, context, args, kws):
        from numba import typing
        # TODO: match self.atypes with args
        return typing.signature(self.rtype, *self.atypes)
        
    def get_call_signatures(self):
        # see explain_function_type in numba/typing/context.py
        # will be used when FunctionType is derived from Callable
        print(f'get_call_signatures()')
        #return (), False   
        raise NotImplementedError(f'get_call_signature()')

    def signature(self):
        from numba import typing
        return typing.signature(self.rtype, *self.atypes)
