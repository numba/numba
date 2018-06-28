from __future__ import print_function, division, absolute_import

import traceback
import inspect
import sys

from .abstract import *
from .common import *
from numba.ir import Loc
from numba import errors

# terminal color markup
_termcolor = errors.termcolor()

class _ResolutionFailures(object):
    """Collect and format function resolution failures.
    """
    def __init__(self, context, function_type, args, kwargs):
        self._context = context
        self._function_type = function_type
        self._args = args
        self._kwargs = kwargs
        self._failures = []

    def __len__(self):
        return len(self._failures)

    def add_error(self, calltemplate, error):
        """
        Args
        ----
        calltemplate : CallTemplate
        error : Exception or str
            Error message
        """
        self._failures.append((calltemplate, error))

    def format(self):
        """Return a formatted error message from all the gathered errors.
        """
        indent = ' ' * 4
        args = [str(a) for a in self._args]
        args += ["%s=%s" % (k, v) for k, v in sorted(self._kwargs.items())]
        headtmp = 'Invalid usage of {} with parameters ({})'
        msgbuf = [headtmp.format(self._function_type, ', '.join(args))]
        explain = self._context.explain_function_type(self._function_type)
        msgbuf.append(explain)
        for i, (temp, error) in enumerate(self._failures):
            msgbuf.append("In definition {}:".format(i))
            msgbuf.append(_termcolor.highlight('{}{}'.format(
                indent, self.format_error(error))))
            loc = self.get_loc(temp, error)
            if loc:
                msgbuf.append('{}raised from {}'.format(indent, loc))

        return '\n'.join(msgbuf)

    def format_error(self, error):
        """Format error message or exception
        """
        if isinstance(error, Exception):
            return '{}: {}'.format(type(error).__name__, error)
        else:
            return '{}'.format(error)

    def get_loc(self, classtemplate, error):
        """Get source location information from the error message.
        """
        if isinstance(error, Exception) and hasattr(error, '__traceback__'):
            # traceback is unavailable in py2
            frame = traceback.extract_tb(error.__traceback__)[-1]
            return "{}:{}".format(frame[0], frame[1])


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
        return self.get_call_type_with_literals(context, args, kws,
                                                literals=None)

    def get_call_type_with_literals(self, context, args, kws, literals):
        failures = _ResolutionFailures(context, self, args, kws)
        for temp_cls in self.templates:
            temp = temp_cls(context)
            try:
                if literals is not None and temp.support_literals:
                    sig = temp.apply(*literals)
                else:
                    sig = temp.apply(args, kws)
            except Exception as e:
                sig = None
                failures.add_error(temp_cls, e)
            else:
                if sig is not None:
                    self._impl_keys[sig.args] = temp.get_impl_key(sig)
                    return sig
                else:
                    failures.add_error(temp_cls, "All templates rejected")

        if len(failures) == 0:
            raise AssertionError("Internal Error. "
                                 "Function resolution ended with no failures "
                                 "or successfull signature")

        raise errors.TypingError(failures.format())

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

    def get_call_type_with_literals(self, context, args, kws, literals):
        if literals is not None and self.template.support_literals:
            return self.template(context).apply(*literals)
        else:
            return self.get_call_type(context, args, kws)

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
        super(Dispatcher, self).__init__("type(%s)" % dispatcher)

    def get_call_type(self, context, args, kws):
        """
        Resolve a call to this dispatcher using the given argument types.
        A signature returned and it is ensured that a compiled specialization
        is available for it.
        """
        template, pysig, args, kws = self.dispatcher.get_call_template(args, kws)
        sig = template(context).apply(args, kws)
        if sig:
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


class RecursiveCall(Opaque):
    """
    Recursive call to a Dispatcher.
    """
    _overloads = None

    def __init__(self, dispatcher_type):
        assert isinstance(dispatcher_type, Dispatcher)
        self.dispatcher_type = dispatcher_type
        name = "recursive(%s)" % (dispatcher_type,)
        super(RecursiveCall, self).__init__(name)
        # Initializing for the first time
        if self._overloads is None:
            self._overloads = {}

    @property
    def overloads(self):
        return self._overloads

    @property
    def key(self):
        return self.dispatcher_type
