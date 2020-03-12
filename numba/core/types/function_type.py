
__all__ = ['FunctionType', 'UndefinedFunctionType', 'FunctionPrototype', 'WrapperAddressProtocol']

from abc import ABC, abstractmethod
from .abstract import Type
from .. import types, utils


class FunctionTypeImplBase:
    """Base class for function type implementations.
    """

    @property
    def signature(self):
        """Return the signature of the function type.
        """
        raise NotImplementedError(f'{type(self).__name__}.signature')

    @property
    def nargs(self):
        return len(self.signature.args)

    @property
    def ftype(self):
        sig = self.signature
        return FunctionPrototype(sig.return_type, sig.args)

    @property
    def key(self):
        return self.ftype.key

    @property
    def name(self):
        return f'{type(self).__name__}[{self.key}]'

    def signatures(self):
        """Generator of the function type signatures.
        """
        yield self.signature

    def check_signature(self, sig, compile=False):
        """Check if the given signature belongs to the function type.
        """
        self_sig = self.signature
        if sig == self_sig:
            return True
        if self_sig.return_type is types.undefined:
            return self.nargs == len(sig.args)
        return False

    def get_call_type(self, context, args, kws):
        from numba.core import typing
        from numba.core.typing.templates import Signature
        # use when a new compilation is not feasible
        sig = self.signature
        if len(args) == self.nargs:
            sig_atypes = []
            for atype, sig_atype in zip(args, sig.args):
                atype = types.unliteral(atype)
                if sig_atype is types.undefined:
                    sig_atypes.append(atype)
                    continue
                sig_atypes.append(sig_atype)
                conv_score = context.context.can_convert(
                    fromty=atype, toty=sig_atype
                )
                # Allow safe casts
                if conv_score is None or conv_score > typing.context.Conversion.safe:
                    break
            else:
                return Signature(sig.return_type, tuple(sig_atypes), recvr=None)
        raise ValueError(
            f'{self} argument types do not match with {args}')


class FunctionTypeImpl(FunctionTypeImplBase):
    
    def __init__(self, signature, dispatchers=None):
        self._signature = types.unliteral(signature)
        self.dispatchers = dispatchers or set()

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__}')
        self._signature.dump(tab = tab + '  ')
        print(f'{tab}END DUMP {type(self).__name__}')

    @property
    def signature(self):
        return self._signature

    def has_signatures(self):
        return True

    def __repr__(self):
        dinfo = []
        for dispatcher in self.dispatchers:
            dinfo.append(tuple([cres.signature for cres in dispatcher.overloads.values()]))
        return f'{type(self).__name__}({self.signature})[{dinfo}]'

    def get_call_type(self, context, args, kws):
        # print(f'get_call_type({args=})')
        sig = super(FunctionTypeImpl, self).get_call_type(context, args, kws)
        # print(f'{sig=}')
        if sig.return_type is types.undefined and self.dispatchers:
            for dispatcher in self.dispatchers:
                template, pysig, args, kws = dispatcher.get_call_template(args, kws)
                new_sig = template(context.context).apply(args, kws)
                # print(f'{new_sig=}')
                return types.unliteral(new_sig)
        #if sig.return_type is types.undefined:
        #    raise
        return sig

class CFuncFunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, cfunc):
        self.cfunc = cfunc
        self._signature = types.unliteral(self.cfunc._sig)

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__}')
        self._signature.dump(tab = tab + '  ')
        print(f'{tab}END DUMP {type(self).__name__}')

    @property
    def signature(self):
        return self._signature

    def has_signatures(self):
        return True

    @property
    def ___name(self):
        return f'{type(self).__name__}({self.cfunc._pyfunc.__name__})'


    def __repr__(self):
        return f'{type(self).__name__}({self.cfunc})'
    

    

class __DispatcherFunctionTypeImpl(FunctionTypeImplBase):

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__}')
        self.dispatcher.dump(tab = tab + '  ')
        print(f'{tab}END DUMP {type(self).__name__}')

    @property
    def signature(self):
        # return the signature of the last compile result
        for cres in reversed(self.dispatcher.overloads.values()):
            return types.unliteral(cres.signature)
        # return something that is a signature
        mn, mx = utils.get_nargs_range(self.dispatcher.py_func)
        return types.unknown(*((types.unknown,) * mn))

    def has_signatures(self):
        return len(self.dispatcher.overloads) > 0

    def signatures(self):
        for cres in self.dispatcher.overloads.values():
            yield types.unliteral(cres.signature)

    #@property
    #def name(self):
    #    return f'{type(self).__name__}({self.dispatcher.py_func.__name__})'

    def __repr__(self):
        return f'{self.name}[{", ".join(map(str, self.signatures()))}]'

    def get_call_type(self, context, args, kws):
        args = tuple(map(types.unliteral, args))
        cres = self.dispatcher.get_compile_result(args, compile=False)
        if cres is not None:
            return types.unliteral(cres.signature)
        template, pysig, args, kws = self.dispatcher.get_call_template(args,
                                                                       kws)
        sig = template(context.context).apply(args, kws)
        # return sig  # uncomment to reproduce `a() + b() -> 2 * a()` issue
        return types.unliteral(sig)

    def check_signature(self, sig, compile=False):
        if super(DispatcherFunctionTypeImpl, self).check_signature(
                sig, compile=compile):
            return True
        if compile:
            try:
                self.dispatcher.compile(sig.args)
                return True
            except Exception:
                pass
        return False


class FunctionType(Type):
    """
    First-class function type.
    """

    cconv = None

    def __init__(self, impl):
        self.impl = impl
        self._key = impl.key

    def __str__(self):
        return f'{type(self).__name__}[{self.ftype}]'

    def is_precise(self):
        return self.impl.signature.is_precise()

    def get_function_type(self, sig=None):
        if sig is not None:
            assert self.signature() == sig
        return self
    
    @property
    def has_defined_args(self):
        sig = self.impl.signature
        for atype in sig.args:
            if atype is types.undefined:
                return False
        return True

    @property
    def has_defined_return_type(self):
        return self.impl.signature.return_type is not types.undefined

    @property
    def is_fully_defined(self):
        return self.has_defined_args and self.has_defined_return_type
        
    @property
    def is_polymorphic(self):
        return self.impl.signature.is_polymorphic

    @property
    def nargs(self):
        return self.impl.nargs
    
    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__}[code={self._code}]')
        self.impl.dump(tab = tab + '  ')
        print(f'{tab}END DUMP {type(self).__name__}')

    @staticmethod
    def extract_function_types(obj):
        """Return function types that the given Python object holds as an
        iterator.  Objects holding no function types yield empty iterator.
        """
        # todo: revise
        from numba.core.dispatcher import Dispatcher
        from numba.core.ccallback import CFunc
        if isinstance(obj, CFunc):
            impl = FunctionTypeImpl(types.unliteral(cfunc._sig))
            yield FunctionType(impl)
        elif isinstance(obj, Dispatcher):
            print(f'{type(obj)=} {len(obj.overloads)=}')
            for cres in obj.overloads.values():
                impl = FunctionTypeImpl(cres.signature)
                yield FunctionType(impl)
        elif isinstance(obj, WrapperAddressProtocol):
            impl = FunctionTypeImpl(obj.signature())
            yield FunctionType(impl)

    @staticmethod
    def fromobject(obj):
        from numba.core.dispatcher import Dispatcher
        from numba.core.ccallback import CFunc
        from numba.core.typing.templates import Signature
        if isinstance(obj, CFunc):
            impl = FunctionTypeImpl(types.unliteral(obj._sig))
            #impl = CFuncFunctionTypeImpl(obj)
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

    def get_numba_type(self):
        if isinstance(self.impl, DispatcherFunctionTypeImpl):
            return self.impl.dispatcher._numba_type_
        return self

    def __repr__(self):
        return f'{type(self).__name__}({self.impl!r})'

    @property
    def name(self):
        return self.impl.name

    @property
    def key(self):
        return self._key

    def has_signatures(self):
        """Return True if function type contains known signatures.
        """
        return self.impl.has_signatures()

    def signature(self):
        return self.impl.signature

    @property
    def ftype(self):
        return self.impl.ftype

    def get_call_type(self, context, args, kws):
        return self.impl.get_call_type(context, args, kws)

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

    def check_signature(self, sig, compile=False):
        """Check if the given signature belongs to the function type.
        Try to compile if enabled.
        """
        return self.impl.check_signature(sig, compile=compile)

    def matches(self, other, compile=False):
        """Return True if other's signatures matches exactly with self's
        signatures. If other does not have known signatures, try to
        compile (if enabled).
        """
        if self.has_signatures():
            if other.has_signatures():
                return self.signature() == other.signature()
            if compile:
                return other.check_signature(self.signature(), compile=compile)
        return False

    def unify(self, context, other):
        if isinstance(other, types.FunctionType) and self.nargs == other.nargs:
            if other.signature().return_type is types.undefined:
                return self


class UndefinedFunctionType(FunctionType):

    _counter = 0

    def __init__(self, impl):
        super(UndefinedFunctionType, self).__init__(impl)

        type(self)._counter += 1
        self._key += str(type(self)._counter)

    @property
    def dispatchers(self):
        return self.impl.dispatchers

    @classmethod
    def make(cls, nargs, dispatchers=None):
        from numba.core.typing.templates import Signature
        sig = Signature(types.undefined, (types.undefined,) * nargs, recvr=None)
        impl = FunctionTypeImpl(sig, dispatchers=dispatchers)
        undefined_function_type = cls(impl)
        return undefined_function_type

    def get_function_type(self, sig=None):
        """
        Return defined function type if possible.
        """
        # print(f'GET FUNCTION TYPE: {sig=}')
        if sig is None:
            for dispatcher in self.dispatchers:
                for cres in dispatcher.overloads.values():
                    sig = types.unliteral(cres.signature)
                    break
                else:
                    continue
                break
        if sig is None:
            return self
        impl = FunctionTypeImpl(sig)
        self.impl = impl
        return FunctionType(impl)


class FunctionPrototype(Type):
    """
    Represents the prototype of a first-class function type.
    Used internally.
    """
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


class CompileResultWAP(WrapperAddressProtocol):

    def __init__(self, cres):
        self.cres = cres
        self.address = cres.get_addresses()['cfunc_wrapper']

    def dump(self, tab=''):
        print(f'{tab}DUMP {type(self).__name__} [addr={self.address}]')
        self.cres.signature.dump(tab=tab + '  ')
        print(f'{tab}END DUMP {type(self).__name__}')

    def __wrapper_address__(self):
        return self.address

    def signature(self):
        return self.cres.signature

    def __call__(self, *args, **kwargs):  # used in object-mode
        return self.cres.entry_point(*args, **kwargs)
