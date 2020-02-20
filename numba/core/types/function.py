
__all__ = ['FunctionType', 'FunctionPrototype', 'WrapperAddressProtocol']

from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, utils


class FunctionTypeImplBase:
    """Base class for function type implementations.
    """

    def signature(self):
        """Return the signature of the function type.
        """
        raise NotImplementedError(f'{type(self).__name__}.signature')

    def signatures(self):
        """Generator of the function type signatures.
        """
        yield self.signature

    def check_signature(self, sig):
        """Check if the given signature belongs to the function type.
        """
        for signature in self.signatures():
            if sig == signature:
                return True
        return False

    def get_call_type(self, context, args, kws):
        # use when a new compilation is not feasible
        for signature in self.signatures():
            if len(args) != len(signature.args):
                continue
            for atype, sig_atype in zip(args, signature.args):
                atype = types.unliteral(atype)
                if isinstance(atype, type(sig_atype)):
                    continue
                elif (isinstance(atype, types.Number)
                      and type(atype) is type(sig_atype)
                      and atype <= sig_atype):
                    continue
                elif (isinstance(atype, types.Array)
                      and isinstance(sig_atype, types.Array)
                      and type(atype.dtype) is type(sig_atype.dtype)
                      # noqa: E721
                      and atype.dtype <= sig_atype.dtype):
                    continue
                else:
                    break
            else:
                # TODO: there may be better matches
                return signature
        raise ValueError(
            f'{self} argument types do not match with {args}')


class FunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, signature):
        self.signature = types.unliteral(signature)

    @property
    def name(self):
        return f'{type(self).__name__}'

    def __repr__(self):
        return f'{self.name}[{self.signature}]'


class CFuncFunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, cfunc):
        self.cfunc = cfunc
        self.signature = types.unliteral(self.cfunc._sig)

    @property
    def name(self):
        return f'{type(self).__name__}({self.cfunc._pyfunc.__name__})'

    def __repr__(self):
        return f'{self.name}[{self.signature}]'


class DispatcherFunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    @property
    def signature(self):
        # return the signature of the last compile result
        for cres in reversed(self.dispatcher.overloads.values()):
            return types.unliteral(cres.signature)
        # return something that is a signature
        mn, mx = utils.get_nargs_range(self.dispatcher.py_func)
        return types.unknown(*((types.unknown,) * mn))

    def signatures(self):
        for cres in self.dispatcher.overloads.values():
            yield types.unliteral(cres.signature)

    @property
    def name(self):
        return f'{type(self).__name__}({self.dispatcher.py_func.__name__})'

    def __repr__(self):
        return f'{self.name}[{", ".join(map(str, self.signatures()))}]'

    def get_call_type(self, context, args, kws):
        # TODO: check if signature exists
        template, pysig, args, kws = self.dispatcher.get_call_template(args,
                                                                       kws)
        sig = template(context.context).apply(args, kws)
        # return sig  # uncomment to reproduce `a() + b() -> 2 * a()` issue
        return types.unliteral(sig)

    def check_signature(self, sig):
        if super(DispatcherFunctionTypeImpl, self).check_signature(sig):
            return True
        try:
            self.dispatcher.compile(sig.args)
        except Exception:
            return False
        return True


class FunctionType(Type):
    """
    First-class function type.
    """

    mutable = True  # What is this for?
    cconv = None

    def __init__(self, impl):
        self.impl = impl

    @staticmethod
    def fromobject(obj):
        from numba.core.dispatcher import Dispatcher
        from numba.core.ccallback import CFunc
        from numba.core.typing.templates import Signature
        if isinstance(obj, CFunc):
            impl = CFuncFunctionTypeImpl(obj)
        elif isinstance(obj, Signature):
            impl = FunctionTypeImpl(obj)
        elif isinstance(obj, Dispatcher):
            if obj.targetoptions.get('no_cfunc_wrapper', True):
                # first-class function is disabled for the given
                # dispatcher instance, fallback to default:
                return types.Dispatcher(obj)
            impl = DispatcherFunctionTypeImpl(obj)
        elif isinstance(obj, WrapperAddressProtocol):
            impl = FunctionTypeImpl(obj.signature())
        else:
            raise NotImplementedError(
                f'function type from {type(obj).__name__}')
        return FunctionType(impl)

    @property
    def name(self):
        return self.ftype.name

    @property
    def key(self):
        return self.name

    def signature(self):
        return self.impl.signature

    @property
    def ftype(self):
        sig = self.signature()
        return FunctionPrototype(sig.return_type, sig.args)

    def get_call_type(self, context, args, kws):
        return self.impl.get_call_type(context, args, kws)

    def __call__(self, *args, **kwargs):  ## remove?
        # defined here to enable resolve_function_type call
        raise NotImplementedError(f'{type(self).__name__}.__call__')

    def resolve_function_type(self, func_type, args, kws):  ## remove?
        raise NotImplementedError(repr((func_type, args, kws)))

    def get_ftype(self, sig=None):
        """Return function prorotype of the given or default signature.
        """
        if sig is None:
            sig = self.signature()
        else:
            # Check that the first-class function type implementation
            # supports the given signature.
            if not self.impl.check_signature(sig):
                raise ValueError(f'{repr(self)} does not support {sig}')
        return FunctionPrototype(sig.return_type, sig.args)


class FunctionPrototype(Type):
    """
    Represents the prototype of a first-class function type.
    Used internally.
    """
    mutable = True
    cconv = None

    def __init__(self, rtype, atypes):
        self.rtype = rtype
        self.atypes = tuple(atypes)

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
    as arguments to Numba jit compiled functions where it can be used
    as first-class functions. As a minimum, the derived types must
    implement two methods ``__wrapper_address__`` and ``signature``.
    """

    @abstractmethod
    def __wrapper_address__(self):
        """Return the address of a first-class function.

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
