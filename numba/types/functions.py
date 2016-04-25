from __future__ import print_function, division, absolute_import

from .abstract import *
from .common import *


class BaseFunction(Callable):
    """
    Base type class for some function types.
    """

    def __init__(self, template):
        if isinstance(template, (list, tuple)):
            self.templates = tuple(template)
            keys = set(temp.key for temp in self.templates)
            if len(keys) != 1:
                raise ValueError("incompatible templates: keys = %s"
                                 % (this,))
            self.typing_key, = keys
        else:
            self.templates = (template,)
            self.typing_key = template.key
        self._impl_keys = {}
        name = "%s(%s)" % (self.__class__.__name__, self.typing_key)
        super(BaseFunction, self).__init__(name)

    @property
    def key(self):
        return self.typing_key, self.templates

    def augment(self, other):
        """
        Augment this function type with the other function types' templates,
        so as to support more input types.
        """
        if type(other) is type(self) and other.typing_key == self.typing_key:
            return type(self)(self.templates + other.templates)

    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
        return self._impl_keys[sig.args]

    def get_call_type(self, context, args, kws):
        for temp_cls in self.templates:
            temp = temp_cls(context)
            sig = temp.apply(args, kws)
            if sig is not None:
                self._impl_keys[sig.args] = temp.get_impl_key(sig)
                return sig

    def get_call_signatures(self):
        sigs = []
        is_param = False
        for temp in self.templates:
            sigs += getattr(temp, 'cases', [])
            is_param = is_param or hasattr(temp, 'generic')
        return sigs, is_param


class Function(BaseFunction, Opaque):
    """
    Type class for builtin functions implemented by Numba.
    """


class BoundFunction(Callable, Opaque):
    """
    A function with an implicit first argument (denoted as *this* below).
    """

    def __init__(self, template, this):
        # Create a derived template with an attribute *this*
        newcls = type(template.__name__ + '.' + str(this), (template,),
                      dict(this=this))
        self.template = newcls
        self.typing_key = self.template.key
        self.this = this
        name = "%s(%s for %s)" % (self.__class__.__name__,
                                  self.typing_key, self.this)
        super(BoundFunction, self).__init__(name)

    def unify(self, typingctx, other):
        if (isinstance(other, BoundFunction) and
            self.typing_key == other.typing_key):
            this = typingctx.unify_pairs(self.this, other.this)
            if this is not None:
                # XXX is it right that both template instances are distinct?
                return self.copy(this=this)

    def copy(self, this):
        return type(self)(self.template, this)

    @property
    def key(self):
        return self.typing_key, self.this

    def get_impl_key(self, sig):
        """
        Get the implementation key (used by the target context) for the
        given signature.
        """
        return self.typing_key

    def get_call_type(self, context, args, kws):
        return self.template(context).apply(args, kws)

    def get_call_signatures(self):
        sigs = getattr(self.template, 'cases', [])
        is_param = hasattr(self.template, 'generic')
        return sigs, is_param


class WeakType(Type):
    """
    Base class for types parametered by a mortal object, to which only
    a weak reference is kept.
    """

    def _store_object(self, obj):
        self._wr = weakref.ref(obj)

    def _get_object(self):
        obj = self._wr()
        if obj is None:
            raise ReferenceError("underlying object has vanished")
        return obj

    @property
    def key(self):
        return self._wr

    def __eq__(self, other):
        if type(self) is type(other):
            obj = self._wr()
            return obj is not None and obj is other._wr()

    def __hash__(self):
        return Type.__hash__(self)


class Dispatcher(WeakType, Callable, Dummy):
    """
    Type class for @jit-compiled functions.
    """

    def __init__(self, dispatcher):
        self._store_object(dispatcher)
        super(Dispatcher, self).__init__("Dispatcher(%s)" % dispatcher)

    def get_call_type(self, context, args, kws):
        """
        Resolve a call to this dispatcher using the given argument types.
        A signature returned and it is ensured that a compiled specialization
        is available for it.
        """
        template, pysig, args, kws = self.dispatcher.get_call_template(args, kws)
        sig = template(context).apply(args, kws)
        sig.pysig = pysig
        return sig

    def get_call_signatures(self):
        sigs = self.dispatcher.nopython_signatures
        return sigs, True

    @property
    def dispatcher(self):
        """
        A strong reference to the underlying numba.dispatcher.Dispatcher instance.
        """
        return self._get_object()

    def get_overload(self, sig):
        """
        Get the compiled overload for the given signature.
        """
        return self.dispatcher.get_overload(sig.args)

    def get_impl_key(self, sig):
        """
        Get the implementation key for the given signature.
        """
        return self.get_overload(sig)


class ExternalFunctionPointer(BaseFunction):
    """
    A pointer to a native function (e.g. exported via ctypes or cffi).
    *get_pointer* is a Python function taking an object
    and returning the raw pointer value as an int.
    """
    def __init__(self, sig, get_pointer, cconv=None):
        from ..typing.templates import (AbstractTemplate, make_concrete_template,
                                        signature)
        from . import ffi_forced_object
        if sig.return_type == ffi_forced_object:
            raise TypeError("Cannot return a pyobject from a external function")
        self.sig = sig
        self.requires_gil = any(a == ffi_forced_object for a in self.sig.args)
        self.get_pointer = get_pointer
        self.cconv = cconv
        if self.requires_gil:
            class GilRequiringDefn(AbstractTemplate):
                key = self.sig

                def generic(self, args, kws):
                    if kws:
                        raise TypeError("does not support keyword arguments")
                    # Make ffi_forced_object a bottom type to allow any type to be
                    # casted to it. This is the only place that support
                    # ffi_forced_object.
                    coerced = [actual if formal == ffi_forced_object else formal
                               for actual, formal
                               in zip(args, self.key.args)]
                    return signature(self.key.return_type, *coerced)
            template = GilRequiringDefn
        else:
            template = make_concrete_template("CFuncPtr", sig, [sig])
        super(ExternalFunctionPointer, self).__init__(template)

    @property
    def key(self):
        return self.sig, self.cconv, self.get_pointer


class ExternalFunction(Function):
    """
    A named native function (resolvable by LLVM) accepting an explicit signature.
    For internal use only.
    """

    def __init__(self, symbol, sig):
        from .. import typing
        self.symbol = symbol
        self.sig = sig
        template = typing.make_concrete_template(symbol, symbol, [sig])
        super(ExternalFunction, self).__init__(template)

    @property
    def key(self):
        return self.symbol, self.sig


class NumbaFunction(Function):
    """
    A named native function with the Numba calling convention
    (resolvable by LLVM).
    For internal use only.
    """

    def __init__(self, fndesc, sig):
        from .. import typing
        self.fndesc = fndesc
        self.sig = sig
        template = typing.make_concrete_template(fndesc.qualname,
                                                 fndesc.qualname, [sig])
        super(NumbaFunction, self).__init__(template)

    @property
    def key(self):
        return self.fndesc.unique_name, self.sig


class NamedTupleClass(Callable, Opaque):
    """
    Type class for namedtuple classes.
    """

    def __init__(self, instance_class):
        self.instance_class = instance_class
        name = "class(%s)" % (instance_class)
        super(NamedTupleClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        # Overriden by the __call__ constructor resolution in typing.collections
        return None

    def get_call_signatures(self):
        return (), True

    @property
    def key(self):
        return self.instance_class


class NumberClass(Callable, DTypeSpec, Opaque):
    """
    Type class for number classes (e.g. "np.float64").
    """

    def __init__(self, instance_type):
        self.instance_type = instance_type
        name = "class(%s)" % (instance_type,)
        super(NumberClass, self).__init__(name)

    def get_call_type(self, context, args, kws):
        # Overriden by the __call__ constructor resolution in typing.builtins
        return None

    def get_call_signatures(self):
        return (), True

    @property
    def key(self):
        return self.instance_type

    @property
    def dtype(self):
        return self.instance_type
