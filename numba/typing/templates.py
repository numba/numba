"""
Define typing templates
"""
from __future__ import print_function, division, absolute_import

import functools
import sys
from types import MethodType

from .. import types, utils
from ..errors import TypingError, InternalError

_IS_PY3 = sys.version_info >= (3,)


class Signature(object):
    """
    The signature of a function call or operation, i.e. its argument types
    and return type.
    """

    # XXX Perhaps the signature should be a BoundArguments, instead
    # of separate args and pysig...
    __slots__ = 'return_type', 'args', 'recvr', 'pysig'

    def __init__(self, return_type, args, recvr, pysig=None):
        if isinstance(args, list):
            args = tuple(args)
        self.return_type = return_type
        self.args = args
        self.recvr = recvr
        self.pysig = pysig

    def replace(self, **kwargs):
        """Copy and replace the given attributes provided as keyword arguments.
        Returns an updated copy.
        """
        curstate = dict(return_type=self.return_type,
                        args=self.args,
                        recvr=self.recvr,
                        pysig=self.pysig)
        curstate.update(kwargs)
        return Signature(**curstate)

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
        """
        Whether this signature represents a bound method or a regular
        function.
        """
        return self.recvr is not None

    def as_method(self):
        """
        Convert this signature to a bound method signature.
        """
        if self.recvr is not None:
            return self
        sig = signature(self.return_type, *self.args[1:],
                        recvr=self.args[0])

        # Adjust the python signature
        params = list(self.pysig.parameters.values())[1:]
        sig.pysig = utils.pySignature(
            parameters=params,
            return_annotation=self.pysig.return_annotation,
        )
        return sig

    def as_function(self):
        """
        Convert this signature to a regular function signature.
        """
        if self.recvr is None:
            return self
        sig = signature(self.return_type, *((self.recvr,) + self.args))
        return sig


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
    # Collect args in the right order
    args = tuple(ba.arguments[param.name]
                 for param in pysig.parameters.values())
    return args


class FunctionTemplate(object):
    # Set to true to disable unsafe cast.
    # subclass overide-able
    unsafe_casting = True
    exact_match_required = False

    def __init__(self, context):
        self.context = context

    def _select(self, cases, args, kws):
        options = {
            'unsafe_casting': self.unsafe_casting,
            'exact_match_required': self.exact_match_required,
        }
        selected = self.context.resolve_overload(self.key, cases, args, kws,
                                                 **options)
        return selected

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        # Lookup the key on the class, to avoid binding it with `self`.
        key = type(self).key
        # On Python 2, we must also take care about unbound methods
        if isinstance(key, MethodType):
            assert key.im_self is None
            key = key.im_func
        return key


class AbstractTemplate(FunctionTemplate):
    """
    Defines method ``generic(self, args, kws)`` which compute a possible
    signature base on input types.  The signature does not have to match the
    input types. It is compared against the input types afterwards.
    """

    def apply(self, args, kws):
        generic = getattr(self, "generic")
        sig = generic(args, kws)
        # Enforce that *generic()* must return None or Signature
        if sig is not None:
            if not isinstance(sig, Signature):
                raise AssertionError(
                    "generic() must return a Signature or None. "
                    "{} returned {}".format(generic, type(sig)),
                )

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
            if not isinstance(sig, types.Type):
                raise TypeError("invalid return type for callable template: got %r"
                                % (sig,))
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
        return self._select(cases, args, kws)


class _OverloadFunctionTemplate(AbstractTemplate):
    """
    A base class of templates for overload functions.
    """

    def _validate_sigs(self, typing_func, impl_func):
        # check that the impl func and the typing func have the same signature!
        typing_sig = utils.pysignature(typing_func)
        impl_sig = utils.pysignature(impl_func)
        # the typing signature is considered golden and must be adhered to by
        # the implementation...
        # Things that are valid:
        # 1. args match exactly
        # 2. kwargs match exactly in name and default value
        # 3. Use of *args in the same location by the same name in both typing
        #    and implementation signature
        # 4. Use of *args in the implementation signature to consume any number
        #    of arguments in the typing signature.
        # Things that are invalid:
        # 5. Use of *args in the typing signature that is not replicated
        #    in the implementing signature
        # 6. Use of **kwargs

        def get_args_kwargs(sig):
            kws = []
            args = []
            pos_arg = None
            for x in sig.parameters.values():
                if x.default == utils.pyParameter.empty:
                    args.append(x)
                    if x.kind == utils.pyParameter.VAR_POSITIONAL:
                        pos_arg = x
                    elif x.kind == utils.pyParameter.VAR_KEYWORD:
                        msg = ("The use of VAR_KEYWORD (e.g. **kwargs) is "
                               "unsupported. (offending argument name is '%s')")
                        raise InternalError(msg % x)
                else:
                    kws.append(x)
            return args, kws, pos_arg

        ty_args, ty_kws, ty_pos = get_args_kwargs(typing_sig)
        im_args, im_kws, im_pos = get_args_kwargs(impl_sig)

        sig_fmt = ("Typing signature:         %s\n"
                   "Implementation signature: %s")
        sig_str = sig_fmt % (typing_sig, impl_sig)

        err_prefix = "Typing and implementation arguments differ in "

        a = ty_args
        b = im_args
        if ty_pos:
            if not im_pos:
                # case 5. described above
                msg = ("VAR_POSITIONAL (e.g. *args) argument kind (offending "
                       "argument name is '%s') found in the typing function "
                       "signature, but is not in the implementing function "
                       "signature.\n%s") % (ty_pos, sig_str)
                raise InternalError(msg)
        else:
            if im_pos:
                # no *args in typing but there's a *args in the implementation
                # this is case 4. described above
                b = im_args[:im_args.index(im_pos)]
                try:
                    a = ty_args[:ty_args.index(b[-1]) + 1]
                except ValueError:
                    # there's no b[-1] arg name in the ty_args, something is
                    # very wrong, we can't work out a diff (*args consumes
                    # unknown quantity of args) so just report first error
                    specialized = "argument names.\n%s\nFirst difference: '%s'"
                    msg = err_prefix + specialized % (sig_str, b[-1])
                    raise InternalError(msg)

        if _IS_PY3:
            def gen_diff(typing, implementing):
                diff = set(typing) ^ set(implementing)
                return "Difference: %s" % diff
        else:
            # funcsigs.Parameter cannot be hashed
            def gen_diff(typing, implementing):
                pass

        if a != b:
            specialized = "argument names.\n%s\n%s" % (sig_str, gen_diff(a, b))
            raise InternalError(err_prefix + specialized)

        # ensure kwargs are the same
        ty = [x.name for x in ty_kws]
        im = [x.name for x in im_kws]
        if ty != im:
            specialized = "keyword argument names.\n%s\n%s"
            msg = err_prefix + specialized % (sig_str, gen_diff(ty_kws, im_kws))
            raise InternalError(msg)
        same = [x.default for x in ty_kws] == [x.default for x in im_kws]
        if not same:
            specialized = "keyword argument default values.\n%s\n%s"
            msg = err_prefix + specialized % (sig_str, gen_diff(ty_kws, im_kws))
            raise InternalError(msg)

    def generic(self, args, kws):
        """
        Type the overloaded function by compiling the appropriate
        implementation for the given args.
        """
        cache_key = self.context, tuple(args), tuple(kws.items())
        try:
            disp = self._impl_cache[cache_key]
        except KeyError:
            # Get the overload implementation for the given types
            pyfunc = self._overload_func(*args, **kws)
            if pyfunc is None:
                # No implementation => fail typing
                self._impl_cache[cache_key] = None
                return
            # check that the typing and impl sigs match up
            if self._strict:
                self._validate_sigs(self._overload_func, pyfunc)
            from numba import jit
            jitdecor = jit(nopython=True, **self._jit_options)
            disp = self._impl_cache[cache_key] = jitdecor(pyfunc)
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


def make_overload_template(func, overload_func, jit_options, strict):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    Compiler options are passed as a dictionary to *jit_options*.
    """
    func_name = getattr(func, '__name__', str(func))
    name = "OverloadTemplate_%s" % (func_name,)
    base = _OverloadFunctionTemplate
    dct = dict(key=func, _overload_func=staticmethod(overload_func),
               _impl_cache={}, _compiled_overloads={}, _jit_options=jit_options,
               _strict=strict)
    return type(base)(name, (base,), dct)


class _IntrinsicTemplate(AbstractTemplate):
    """
    A base class of templates for intrinsic definition
    """

    def generic(self, args, kws):
        """
        Type the intrinsic by the arguments.
        """
        from numba.targets.imputils import lower_builtin

        cache_key = self.context, args, tuple(kws.items())
        try:
            return self._impl_cache[cache_key]
        except KeyError:
            result = self._definition_func(self.context, *args, **kws)
            if result is None:
                return
            [sig, imp] = result
            pysig = utils.pysignature(self._definition_func)
            # omit context argument from user function
            parameters = list(pysig.parameters.values())[1:]
            sig.pysig = pysig.replace(parameters=parameters)
            self._impl_cache[cache_key] = sig
            self._overload_cache[sig.args] = imp
            # register the lowering
            lower_builtin(imp, *sig.args)(imp)
            return sig

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        return self._overload_cache[sig.args]


def make_intrinsic_template(handle, defn, name):
    """
    Make a template class for a intrinsic handle *handle* defined by the
    function *defn*.  The *name* is used for naming the new template class.
    """
    base = _IntrinsicTemplate
    name = "_IntrinsicTemplate_%s" % (name)
    dct = dict(key=handle, _definition_func=staticmethod(defn),
               _impl_cache={}, _overload_cache={})
    return type(base)(name, (base,), dct)


class AttributeTemplate(object):
    _initialized = False

    def __init__(self, context):
        self._lazy_class_init()
        self.context = context

    def resolve(self, value, attr):
        return self._resolve(value, attr)

    @classmethod
    def _lazy_class_init(cls):
        if not cls._initialized:
            cls.do_class_init()
            cls._initialized = True

    @classmethod
    def do_class_init(cls):
        """
        Class-wide initialization.  Can be overriden by subclasses to
        register permanent typing or target hooks.
        """

    def _resolve(self, value, attr):
        fn = getattr(self, "resolve_%s" % attr, None)
        if fn is None:
            fn = self.generic_resolve
            if fn is NotImplemented:
                if isinstance(value, types.Module):
                    return self.context.resolve_module_constants(value, attr)
                else:
                    return None
            else:
                return fn(value, attr)
        else:
            return fn(value)

    generic_resolve = NotImplemented


class _OverloadAttributeTemplate(AttributeTemplate):
    """
    A base class of templates for @overload_attribute functions.
    """

    def __init__(self, context):
        super(_OverloadAttributeTemplate, self).__init__(context)
        self.context = context

    @classmethod
    def do_class_init(cls):
        """
        Register attribute implementation.
        """
        from numba.targets.imputils import lower_getattr
        attr = cls._attr

        @lower_getattr(cls.key, attr)
        def getattr_impl(context, builder, typ, value):
            sig_args = (typ,)
            sig_kws = {}
            typing_context = context.typing_context
            disp = cls._get_dispatcher(typing_context, typ, attr, sig_args, sig_kws)
            disp_type = types.Dispatcher(disp)
            sig = disp_type.get_call_type(typing_context, sig_args, sig_kws)
            call = context.get_function(disp_type, sig)
            return call(builder, (value,))

    @classmethod
    def _get_dispatcher(cls, context, typ, attr, sig_args, sig_kws):
        """
        Get the compiled dispatcher implementing the attribute for
        the given formal signature.
        """
        cache_key = context, typ, attr, tuple(sig_args), tuple(sig_kws.items())
        try:
            disp = cls._impl_cache[cache_key]
        except KeyError:
            # Get the overload implementation for the given type
            pyfunc = cls._overload_func(*sig_args, **sig_kws)
            if pyfunc is None:
                # No implementation => fail typing
                cls._impl_cache[cache_key] = None
                return

            from numba import jit
            disp = cls._impl_cache[cache_key] = jit(nopython=True)(pyfunc)
        return disp

    def _resolve_impl_sig(self, typ, attr, sig_args, sig_kws):
        """
        Compute the actual implementation sig for the given formal argument types.
        """
        disp = self._get_dispatcher(self.context, typ, attr, sig_args, sig_kws)
        if disp is None:
            return None

        # Compile and type it for the given types
        disp_type = types.Dispatcher(disp)
        sig = disp_type.get_call_type(self.context, sig_args, sig_kws)
        return sig

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None

        sig = self._resolve_impl_sig(typ, attr, (typ,), {})
        return sig.return_type


class _OverloadMethodTemplate(_OverloadAttributeTemplate):
    """
    A base class of templates for @overload_method functions.
    """

    @classmethod
    def do_class_init(cls):
        """
        Register generic method implementation.
        """
        from numba.targets.imputils import lower_builtin
        attr = cls._attr

        @lower_builtin((cls.key, attr), cls.key, types.VarArg(types.Any))
        def method_impl(context, builder, sig, args):
            typ = sig.args[0]
            typing_context = context.typing_context
            disp = cls._get_dispatcher(typing_context, typ, attr, sig.args, {})
            disp_type = types.Dispatcher(disp)
            sig = disp_type.get_call_type(typing_context, sig.args, {})
            call = context.get_function(disp_type, sig)
            # Link dependent library
            context.add_linking_libs(getattr(call, 'libs', ()))
            return call(builder, args)

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None

        assert isinstance(typ, self.key)

        class MethodTemplate(AbstractTemplate):
            key = (self.key, attr)

            def generic(_, args, kws):
                args = (typ,) + tuple(args)
                sig = self._resolve_impl_sig(typ, attr, args, kws)
                if sig is not None:
                    return sig.as_method()

        return types.BoundFunction(MethodTemplate, typ)


def make_overload_attribute_template(typ, attr, overload_func,
                                     base=_OverloadAttributeTemplate):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, types.Type) or issubclass(typ, types.Type)
    name = "OverloadTemplate_%s_%s" % (typ, attr)
    # Note the implementation cache is subclass-specific
    dct = dict(key=typ, _attr=attr, _impl_cache={},
               _overload_func=staticmethod(overload_func),
               )
    return type(base)(name, (base,), dct)


def make_overload_method_template(typ, attr, overload_func):
    """
    Make a template class for method *attr* of *typ* overloaded by
    *overload_func*.
    """
    return make_overload_attribute_template(typ, attr, overload_func,
                                            base=_OverloadMethodTemplate)


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

    def register_global(self, val=None, typ=None, **kwargs):
        """
        Register the typing of a global value.
        Functional usage with a Numba type::
            register_global(value, typ)

        Decorator usage with a template class::
            @register_global(value, typing_key=None)
            class Template:
                ...
        """
        if typ is not None:
            # register_global(val, typ)
            assert val is not None
            assert not kwargs
            self.globals.append((val, typ))
        else:
            def decorate(cls, typing_key):
                class Template(cls):
                    key = typing_key
                if callable(val):
                    typ = types.Function(Template)
                else:
                    raise TypeError("cannot infer type for global value %r")
                self.globals.append((val, typ))
                return cls

            # register_global(val, typing_key=None)(<template class>)
            assert val is not None
            typing_key = kwargs.pop('typing_key', val)
            assert not kwargs
            if typing_key is val:
                # Check the value is globally reachable, as it is going
                # to be used as the key.
                mod = sys.modules[val.__module__]
                if getattr(mod, val.__name__) is not val:
                    raise ValueError("%r is not globally reachable as '%s.%s'"
                                     % (mod, val.__module__, val.__name__))

            def decorator(cls):
                return decorate(cls, typing_key)
            return decorator


class BaseRegistryLoader(object):
    """
    An incremental loader for a registry.  Each new call to
    new_registrations() will iterate over the not yet seen registrations.

    The reason for this object is multiple:
    - there can be several contexts
    - each context wants to install all registrations
    - registrations can be added after the first installation, so contexts
      must be able to get the "new" installations

    Therefore each context maintains its own loaders for each existing
    registry, without duplicating the registries themselves.
    """

    def __init__(self, registry):
        self._registrations = dict(
            (name, utils.stream_list(getattr(registry, name)))
            for name in self.registry_items)

    def new_registrations(self, name):
        for item in next(self._registrations[name]):
            yield item


class RegistryLoader(BaseRegistryLoader):
    """
    An incremental loader for a typing registry.
    """
    registry_items = ('functions', 'attributes', 'globals')


builtin_registry = Registry()
infer = builtin_registry.register
infer_getattr = builtin_registry.register_attr
infer_global = builtin_registry.register_global
