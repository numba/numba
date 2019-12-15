
__all__ = ['FunctionType', 'FunctionProtoType']

from .abstract import Type


class FunctionType(Type):

    mutable = True
    cconv = None

    def __init__(self, ftype):
        assert isinstance(ftype, FunctionProtoType), type(ftype)
        self.ftype = ftype
        name = ftype.name + '_FUNC'
        super(FunctionType, self).__init__(name)

    def signature(self):
        from numba import typing
        ptype = self.ftype
        return typing.signature(ptype.rtype, *ptype.atypes)

    def get_call_type(self, context, args, kws):
        # TODO: implement match self.ftype.atypes to args
        return self.signature()

    @property
    def key(self):
        return self.name

    def cast_python_value(self, value):
        from numba import typing
        if isinstance(value, typing.Signature):
            ptype = FunctionProtoType(value.return_type, value.args)
            return FunctionType(ptype)
        raise NotImplementedError(
            'cast_python_value({}, {})'.format(value, type(value)))

    def __get_call_type(self, context, args, kws):
        from numba import typing
        # TODO: match self.atypes with args
        return typing.signature(self.rtype, *self.atypes)

    def __get_call_signatures(self):
        # see explain_function_type in numba/typing/context.py
        # will be used when FunctionType is derived from Callable
        # return (), False
        raise NotImplementedError('get_call_signature()')


class FunctionProtoType(Type):
    """
    Represents a first-class function type.
    """
    mutable = True
    cconv = None

    def __init__(self, rtype, atypes):
        from numba.function import mangle
        self.rtype = rtype
        self.atypes = tuple(atypes)
        name = 'FT' + mangle(self)
        super(FunctionProtoType, self).__init__(name)

    @property
    def key(self):
        return self.name
