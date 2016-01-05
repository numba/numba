"""
Define typing templates
"""
from __future__ import print_function, division, absolute_import

import functools
from functools import reduce
import operator

from .. import types, utils
from ..errors import TypingError, UntypedAttributeError


class Signature(object):
    # XXX Perhaps the signature should be a BoundArguments, instead
    # of separate args and pysig...
    __slots__ = 'return_type', 'args', 'recvr', 'pysig'

    def __init__(self, return_type, args, recvr, pysig=None):
        self.return_type = return_type
        self.args = args
        self.recvr = recvr
        self.pysig = pysig

    def __getstate__(self):
        """
        Needed because of __slots__.
        """
        return self.return_type, self.args, self.recvr, self.pysig

    def __setstate__(self, state):
        """
        Needed because of __slots__.
        """
        self.return_type, self.args, self.recvr, self.pysig = state

    def __hash__(self):
        return hash((self.args, self.return_type))

    def __eq__(self, other):
        if isinstance(other, Signature):
            return (self.args == other.args and
                    self.return_type == other.return_type and
                    self.recvr == other.recvr and
                    self.pysig == other.pysig)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s -> %s" % (self.args, self.return_type)

    @property
    def is_method(self):
        return self.recvr is not None


def make_concrete_template(name, key, signatures):
    baseclasses = (ConcreteTemplate,)
    gvars = dict(key=key, cases=list(signatures))
    return type(name, baseclasses, gvars)


def make_callable_template(key, typer, recvr=None):
    """
    Create a callable template with the given key and typer function.
    """
    def generic(self):
        return typer

    name = "%s_CallableTemplate" % (key,)
    bases = (CallableTemplate,)
    class_dict = dict(key=key, generic=generic, recvr=recvr)
    return type(name, bases, class_dict)


def signature(return_type, *args, **kws):
    recvr = kws.pop('recvr', None)
    assert not kws
    return Signature(return_type, args, recvr=recvr)


def fold_arguments(pysig, args, kws, normal_handler, default_handler,
                   stararg_handler):
    """
    Given the signature *pysig*, explicit *args* and *kws*, resolve
    omitted arguments and keyword arguments. A tuple of positional
    arguments is returned.
    Various handlers allow to process arguments:
    - normal_handler(index, param, value) is called for normal arguments
    - default_handler(index, param, default) is called for omitted arguments
    - stararg_handler(index, param, values) is called for a "*args" argument
    """
    ba = pysig.bind(*args, **kws)
    defargs = []
    for i, param in enumerate(pysig.parameters.values()):
        name = param.name
        default = param.default
        if param.kind == param.VAR_POSITIONAL:
            # stararg may be omitted, in which case its "default" value
            # is simply the empty tuple
            ba.arguments[name] = stararg_handler(i, param,
                                                 ba.arguments.get(name, ()))
        elif name in ba.arguments:
            # Non-stararg, present
            ba.arguments[name] = normal_handler(i, param, ba.arguments[name])
        else:
            # Non-stararg, omitted
            assert default is not param.empty
            ba.arguments[name] = default_handler(i, param, default)
    if ba.kwargs:
        # There's a remaining keyword argument, e.g. if omitting
        # some argument with a default value before it.
        raise NotImplementedError("unhandled keyword argument: %s"
                                  % list(ba.kwargs))
    # Collect args in the right order
    args = tuple(ba.arguments[param.name]
                 for param in pysig.parameters.values())
    return args


def _uses_downcast(dists):
    for d in dists:
        if d < 0:
            return True
    return False


def _sum_downcast(dists):
    c = 0
    for d in dists:
        if d < 0:
            c += abs(d)
    return c


class FunctionTemplate(object):
    def __init__(self, context):
        self.context = context

    def _select(self, cases, args, kws):
        selected = self.context.resolve_overload(self.key, cases, args, kws)
        return selected

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        # Lookup the key on the type, to avoid binding it with `self`.
        return type(self).key


class AbstractTemplate(FunctionTemplate):
    """
    Defines method ``generic(self, args, kws)`` which compute a possible
    signature base on input types.  The signature does not have to match the
    input types. It is compared against the input types afterwards.
    """

    def apply(self, args, kws):
        generic = getattr(self, "generic")
        sig = generic(args, kws)

        # Unpack optional type if no matching signature
        if not sig and any(isinstance(x, types.Optional) for x in args):
            def unpack_opt(x):
                if isinstance(x, types.Optional):
                    return x.type
                else:
                    return x

            args = list(map(unpack_opt, args))
            assert not kws  # Not supported yet
            sig = generic(args, kws)

        return sig


class CallableTemplate(FunctionTemplate):
    """
    Base class for a template defining a ``generic(self)`` method
    returning a callable to be called with the actual ``*args`` and
    ``**kwargs`` representing the call signature.  The callable has
    to return a return type, a full signature, or None.  The signature
    does not have to match the input types. It is compared against the
    input types afterwards.
    """
    recvr = None

    def apply(self, args, kws):
        generic = getattr(self, "generic")
        typer = generic()
        sig = typer(*args, **kws)

        # Unpack optional type if no matching signature
        if sig is None:
            if any(isinstance(x, types.Optional) for x in args):
                def unpack_opt(x):
                    if isinstance(x, types.Optional):
                        return x.type
                    else:
                        return x

                args = list(map(unpack_opt, args))
                sig = typer(*args, **kws)
            if sig is None:
                return

        # Get the pysig
        try:
            pysig = typer.pysig
        except AttributeError:
            pysig = utils.pysignature(typer)

        # Fold any keyword arguments
        bound = pysig.bind(*args, **kws)
        if bound.kwargs:
            raise TypingError("unsupported call signature")
        if not isinstance(sig, Signature):
            # If not a signature, `sig` is assumed to be the return type
            assert isinstance(sig, types.Type)
            sig = signature(sig, *bound.args)
        if self.recvr is not None:
            sig.recvr = self.recvr
        # Hack any omitted parameters out of the typer's pysig,
        # as lowering expects an exact match between formal signature
        # and actual args.
        if len(bound.args) < len(pysig.parameters):
            parameters = list(pysig.parameters.values())[:len(bound.args)]
            pysig = pysig.replace(parameters=parameters)
        sig.pysig = pysig
        cases = [sig]
        return self._select(cases, bound.args, bound.kwargs)


class ConcreteTemplate(FunctionTemplate):
    """
    Defines attributes "cases" as a list of signature to match against the
    given input types.
    """

    def apply(self, args, kws):
        cases = getattr(self, 'cases')
        assert cases
        return self._select(cases, args, kws)


class _OverloadFunctionTemplate(AbstractTemplate):
    """
    A base class of templates for overload functions.
    """

    def generic(self, args, kws):
        """
        Type the overloaded function by compiling the appropriate
        implementation for the given args.
        """
        cache_key = self.context, args, tuple(kws.items())
        try:
            disp = self._impl_cache[cache_key]
        except KeyError:
            # Get the overload implementation for the given types
            pyfunc = self._overload_func(*args, **kws)
            if pyfunc is None:
                # No implementation => fail typing
                self._impl_cache[cache_key] = None
                return
            from numba import jit
            disp = self._impl_cache[cache_key] = jit(nopython=True)(pyfunc)
        else:
            if disp is None:
                return

        # Compile and type it for the given types
        disp_type = types.Dispatcher(disp)
        sig = disp_type.get_call_type(self.context, args, kws)
        # Store the compiled overload for use in the lowering phase
        self._compiled_overloads[sig.args] = disp_type.get_overload(sig)
        return sig

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        return self._compiled_overloads[sig.args]


def make_overload_template(func, overload_func):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    """
    func_name = getattr(func, '__name__', str(func))
    name = "OverloadTemplate_%s" % (func_name,)
    base = _OverloadFunctionTemplate
    dct = dict(key=func, _overload_func=staticmethod(overload_func),
               _impl_cache={}, _compiled_overloads={})
    return type(base)(name, (base,), dct)


class AttributeTemplate(object):
    def __init__(self, context):
        self.context = context

    def resolve(self, value, attr):
        return self._resolve(value, attr)

    def _resolve(self, value, attr):
        fn = getattr(self, "resolve_%s" % attr, None)
        if fn is None:
            fn = self.generic_resolve
            if fn is NotImplemented:
                return self.context.resolve_module_constants(value, attr)
            else:
                return fn(value, attr)
        else:
            return fn(value)

    generic_resolve = NotImplemented


class _OverloadAttributeTemplate(AttributeTemplate):
    """
    A base class of templates for @overload_attribute functions.
    """

    def _resolve(self, typ, attr):
        # Attribute-specific overload
        if self._attr == attr:
            cache_key = self.context, typ, attr
            try:
                disp = self._impl_cache[cache_key]
            except KeyError:
                # Get the overload implementation for the given type
                pyfunc = self._overload_func(typ)
                if pyfunc is None:
                    # No implementation => fail typing
                    self._impl_cache[cache_key] = None
                    return

                from numba import jit
                from numba.targets.imputils import impl_attribute, builtin_attr

                disp = self._impl_cache[cache_key] = jit(nopython=True)(pyfunc)

                # Register an implementation on the target(s), calling
                # the compiled user-supplied function
                @builtin_attr
                @impl_attribute(typ, attr)
                def getattr_impl(context, builder, typ, value):
                    disp_type = types.Dispatcher(disp)
                    # `sig` is computed below
                    call = context.get_function(disp_type, sig)
                    return call(builder, (value,))
            else:
                if disp is None:
                    return

            # Compile and type it for the given types
            disp_type = types.Dispatcher(disp)
            sig = disp_type.get_call_type(self.context, (typ,), {})
            return sig.return_type


def make_overload_attribute_template(typ, attr, overload_func):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadTemplate_%s_%s" % (typ, attr)
    base = _OverloadAttributeTemplate
    dct = dict(key=typ, _attr=attr, _impl_cache={},
               _overload_func=staticmethod(overload_func),
               )
    return type(base)(name, (base,), dct)


def bound_function(template_key):
    """
    Wrap an AttributeTemplate resolve_* method to allow it to
    resolve an instance method's signature rather than a instance attribute.
    The wrapped method must return the resolved method's signature
    according to the given self type, args, and keywords.

    It is used thusly:

        class ComplexAttributes(AttributeTemplate):
            @bound_function("complex.conjugate")
            def resolve_conjugate(self, ty, args, kwds):
                return ty

    *template_key* (e.g. "complex.conjugate" above) will be used by the
    target to look up the method's implementation, as a regular function.
    """
    def wrapper(method_resolver):
        @functools.wraps(method_resolver)
        def attribute_resolver(self, ty):
            class MethodTemplate(AbstractTemplate):
                key = template_key
                def generic(_, args, kws):
                    sig = method_resolver(self, ty, args, kws)
                    if sig is not None and sig.recvr is None:
                        sig.recvr = ty
                    return sig

            return types.BoundFunction(MethodTemplate, ty)
        return attribute_resolver
    return wrapper


class MacroTemplate(object):
    pass


# -----------------------------

class Registry(object):
    """
    A registry of typing declarations.  The registry stores such declarations
    for functions, attributes and globals.
    """

    def __init__(self):
        self.functions = []
        self.attributes = []
        self.globals = []

    def register(self, item):
        assert issubclass(item, FunctionTemplate)
        self.functions.append(item)
        return item

    def register_attr(self, item):
        assert issubclass(item, AttributeTemplate)
        self.attributes.append(item)
        return item

    def register_global(self, v, t):
        self.globals.append((v, t))

    def resolves_global(self, global_value, wrapper_type=types.Function,
                        typing_key=None):
        """
        Decorate a FunctionTemplate subclass so that it gets registered
        as resolving *global_value* with the *wrapper_type* (by default
        a types.Function).

        Example use::
            @resolves_global(math.fabs)
            class Math(ConcreteTemplate):
                cases = [signature(types.float64, types.float64)]
        """
        if typing_key is None:
            typing_key = global_value
        def decorate(cls):
            class Template(cls):
                key = typing_key
            self.register_global(global_value, wrapper_type(Template))
            return cls
        return decorate


class RegistryLoader(object):
    """
    An incremental loader for a registry.  Each new call to new_functions(),
    etc. will iterate over the not yet seen registrations.

    The reason for this object is multiple:
    - there can be several contexts
    - each context wants to install all registrations
    - registrations can be added after the first installation, so contexts
      must be able to get the "new" installations

    Therefore each context maintains its own loaders for each existing
    registry, without duplicating the registries themselves.
    """

    def __init__(self, registry):
        self._functions = utils.stream_list(registry.functions)
        self._attributes = utils.stream_list(registry.attributes)
        self._globals = utils.stream_list(registry.globals)

    def new_functions(self):
        for item in next(self._functions):
            yield item

    def new_attributes(self):
        for item in next(self._attributes):
            yield item

    def new_globals(self):
        for item in next(self._globals):
            yield item


builtin_registry = Registry()
builtin = builtin_registry.register
builtin_attr = builtin_registry.register_attr
builtin_global = builtin_registry.register_global
