
__all__ = ['FunctionType', 'FunctionPrototype', 'WrapperAddressProtocol']

from abc import ABC, abstractmethod
from .abstract import Type, Literal


class FunctionType(Type):
    """The front-end for first-class function types.
    """

    mutable = True
    cconv = None

    def __init__(self, ftype):
        if isinstance(ftype, tuple) and len(ftype) == 2:
            ftype = FunctionPrototype(ftype[0], ftype[1])
        if isinstance(ftype, FunctionPrototype):
            self.ftype = ftype
            name = ftype.name
        else:
            raise TypeError(
                f'constructing {type(self).__name__} from a {type(ftype)}'
                ' instance is not supported')
        super(FunctionType, self).__init__(name)

    def signature(self):
        from numba import typing
        ptype = self.ftype
        return typing.signature(ptype.rtype, *ptype.atypes)

    @property
    def key(self):
        return self.name

    def cast_python_value(self, value):
        from numba import typing
        if isinstance(value, typing.Signature):
            ptype = FunctionPrototype(value.return_type, value.args)
            return FunctionType(ptype)
        raise NotImplementedError(
            'cast_python_value({}, {})'.format(value, type(value)))

    def get_call_type(self, context, args, kws):
        from numba import typing
        ptype = self.ftype
        if len(args) == len(ptype.atypes):
            for i, atype in enumerate(args):
                if isinstance(atype, Literal):
                    atype = atype.literal_type
                if not (atype <= ptype.atypes[i]):
                    break
            else:
                return typing.signature(ptype.rtype, *ptype.atypes)
            # TODO: implement overload support
        raise ValueError(
            f'{self} argument types do not match with {args}')


class FunctionPrototype(Type):
    """
    Represents a first-class function type.
    """
    mutable = True
    cconv = None

    def __init__(self, rtype, atypes):
        from numba.types import void
        self.rtype = rtype
        atypes = tuple(atypes)
        if len(atypes) == 1 and atypes[0] == void:
            atypes = ()
        self.atypes = atypes

        assert isinstance(rtype, Type), (rtype)
        lst = []
        for atype in self.atypes:
            assert isinstance(atype, Type), (atype)
            lst.append(atype.name)
        name = '%s(%s)' % (rtype, ', '.join(lst))

        super(FunctionPrototype, self).__init__(name)

    @property
    def key(self):
        return self.name


class WrapperAddressProtocol(ABC):
    """Base class for Wrapper Address Protocol.

    Objects that inherit from the WrapperAddressProtocol can be passed
    as arguments to Numba njit compiled functions where it can be used
    as first-class functions. As a minimum, the derived types must
    implement the two methods __wrapper_address__ and signature.

    """

    @abstractmethod
    def __wrapper_address__(self, sig):
        """Return the address of a first-class function with given signature.

        Parameters
        ----------
        sig : Signature
          A function signature

        Returns
        -------
        addr : int

        """

    @abstractmethod
    def signature(self):
        """Return the signature of a first-class function.

        Returns
        -------
        sig : Signature
          The returned Signature instance represents the type of a
          first-class function that the given WrapperAddressProtocol
          instance represents.
        """
