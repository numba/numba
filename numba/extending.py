
import inspect
import uuid
import weakref
import collections

from numba import types, config

# Exported symbols
from .typing.typeof import typeof_impl
from .typing.templates import infer, infer_getattr
from .targets.imputils import (
    lower_builtin, lower_getattr, lower_getattr_generic,
    lower_setattr, lower_setattr_generic, lower_cast)
from .datamodel import models, register_default as register_model
from .pythonapi import box, unbox, reflect, NativeValue
from ._helperlib import _import_cython_function

def type_callable(func):
    """
    Decorate a function as implementing typing for the callable *func*.
    *func* can be a callable object (probably a global) or a string
    denoting a built-in operation (such 'getitem' or '__array_wrap__')
    """
    from .typing.templates import CallableTemplate, infer, infer_global
    if not callable(func) and not isinstance(func, str):
        raise TypeError("`func` should be a function or string")
    try:
        func_name = func.__name__
    except AttributeError:
        func_name = str(func)

    def decorate(typing_func):
        def generic(self):
            return typing_func(self.context)

        name = "%s_CallableTemplate" % (func_name,)
        bases = (CallableTemplate,)
        class_dict = dict(key=func, generic=generic)
        template = type(name, bases, class_dict)
        infer(template)
        if hasattr(func, '__module__'):
            infer_global(func, types.Function(template))

    return decorate


# By default, an *overload* does not have a cpython wrapper because it is not
# callable from python.
_overload_default_jit_options = {'no_cpython_wrapper': True}


def overload(func, jit_options={}):
    """
    A decorator marking the decorated function as typing and implementing
    *func* in nopython mode.

    The decorated function will have the same formal parameters as *func*
    and be passed the Numba types of those parameters.  It should return
    a function implementing *func* for the given types.

    Here is an example implementing len() for tuple types::

        @overload(len)
        def tuple_len(seq):
            if isinstance(seq, types.BaseTuple):
                n = len(seq)
                def len_impl(seq):
                    return n
                return len_impl

    Compiler options can be passed as an dictionary using the **jit_options**
    argument.
    """
    from .typing.templates import make_overload_template, infer_global

    # set default options
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)  # let user options override

    def decorate(overload_func):
        template = make_overload_template(func, overload_func, opts)
        infer(template)
        if hasattr(func, '__module__'):
            infer_global(func, types.Function(template))
        return overload_func

    return decorate


def register_jitable(*args, **kwargs):
    """
    Register a regular python function that can be executed by the python
    interpreter and can be compiled into a nopython function when referenced
    by other jit'ed functions.  Can be used as::

        @register_jitable
        def foo(x, y):
            return x + y

    Or, with compiler options::

        @register_jitable(_nrt=False) # disable runtime allocation
        def foo(x, y):
            return x + y

    """
    def wrap(fn):
        # It is just a wrapper for @overload
        @overload(fn, jit_options=kwargs)
        def ov_wrap(*args, **kwargs):
            return fn
        return fn

    if kwargs:
        return wrap
    else:
        return wrap(*args)


def overload_attribute(typ, attr):
    """
    A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode.

    Here is an example implementing .nbytes for array types::

        @overload_attribute(types.Array, 'nbytes')
        def array_nbytes(arr):
            def get(arr):
                return arr.size * arr.itemsize
            return get
    """
    # TODO implement setters
    from .typing.templates import make_overload_attribute_template

    def decorate(overload_func):
        template = make_overload_attribute_template(typ, attr, overload_func)
        infer_getattr(template)
        return overload_func

    return decorate


def overload_method(typ, attr):
    """
    A decorator marking the decorated function as typing and implementing
    attribute *attr* for the given Numba type in nopython mode.

    Here is an example implementing .take() for array types::

        @overload_method(types.Array, 'take')
        def array_take(arr, indices):
            if isinstance(indices, types.Array):
                def take_impl(arr, indices):
                    n = indices.shape[0]
                    res = np.empty(n, arr.dtype)
                    for i in range(n):
                        res[i] = arr[indices[i]]
                    return res
                return take_impl
    """
    from .typing.templates import make_overload_method_template

    def decorate(overload_func):
        template = make_overload_method_template(typ, attr, overload_func)
        infer_getattr(template)
        return overload_func

    return decorate


def make_attribute_wrapper(typeclass, struct_attr, python_attr):
    """
    Make an automatic attribute wrapper exposing member named *struct_attr*
    as a read-only attribute named *python_attr*.
    The given *typeclass*'s model must be a StructModel subclass.
    """
    from .typing.templates import AttributeTemplate
    from .datamodel import default_manager
    from .datamodel.models import StructModel
    from .targets.imputils import impl_ret_borrowed
    from . import cgutils

    if not isinstance(typeclass, type) or not issubclass(typeclass, types.Type):
        raise TypeError("typeclass should be a Type subclass, got %s"
                        % (typeclass,))

    def get_attr_fe_type(typ):
        """
        Get the Numba type of member *struct_attr* in *typ*.
        """
        model = default_manager.lookup(typ)
        if not isinstance(model, StructModel):
            raise TypeError("make_struct_attribute_wrapper() needs a type "
                            "with a StructModel, but got %s" % (model,))
        return model.get_member_fe_type(struct_attr)

    @infer_getattr
    class StructAttribute(AttributeTemplate):
        key = typeclass

        def generic_resolve(self, typ, attr):
            if attr == python_attr:
                return get_attr_fe_type(typ)

    @lower_getattr(typeclass, python_attr)
    def struct_getattr_impl(context, builder, typ, val):
        val = cgutils.create_struct_proxy(typ)(context, builder, value=val)
        attrty = get_attr_fe_type(typ)
        attrval = getattr(val, struct_attr)
        return impl_ret_borrowed(context, builder, attrty, attrval)


class _Intrinsic(object):
    """
    Dummy callable for intrinsic
    """
    _memo = weakref.WeakValueDictionary()
    # hold refs to last N functions deserialized, retaining them in _memo
    # regardless of whether there is another reference
    _recent = collections.deque(maxlen=config.FUNCTION_CACHE_SIZE)

    __uuid = None

    def __init__(self, name, defn, support_literals=False):
        self._name = name
        self._defn = defn
        self._support_literals = support_literals

    @property
    def _uuid(self):
        """
        An instance-specific UUID, to avoid multiple deserializations of
        a given instance.

        Note this is lazily-generated, for performance reasons.
        """
        u = self.__uuid
        if u is None:
            u = str(uuid.uuid1())
            self._set_uuid(u)
        return u

    def _set_uuid(self, u):
        assert self.__uuid is None
        self.__uuid = u
        self._memo[u] = self
        self._recent.append(self)

    def _register(self):
        from .typing.templates import make_intrinsic_template, infer_global

        template = make_intrinsic_template(self, self._defn, self._name)
        template.support_literals = self._support_literals
        infer(template)
        infer_global(self, types.Function(template))

    def __call__(self, *args, **kwargs):
        """
        This is only defined to pretend to be a callable from CPython.
        """
        msg = '{0} is not usable in pure-python'.format(self)
        raise NotImplementedError(msg)

    def __repr__(self):
        return "<intrinsic {0}>".format(self._name)

    def __reduce__(self):
        from numba import serialize

        def reduce_func(fn):
            gs = serialize._get_function_globals_for_reduction(fn)
            return serialize._reduce_function(fn, gs)

        return (serialize._rebuild_reduction,
                (self.__class__, str(self._uuid), self._name,
                 reduce_func(self._defn)))

    @classmethod
    def _rebuild(cls, uuid, name, defn_reduced):
        from numba import serialize

        try:
            return cls._memo[uuid]
        except KeyError:
            defn = serialize._rebuild_function(*defn_reduced)

            llc = cls(name=name, defn=defn)
            llc._register()
            llc._set_uuid(uuid)
            return llc


def intrinsic(*args, **kwargs):
    """
    A decorator marking the decorated function as typing and implementing
    *func* in nopython mode using the llvmlite IRBuilder API.  This is an escape
    hatch for expert users to build custom LLVM IR that will be inlined to
    the caller.

    The first argument to *func* is the typing context.  The rest of the
    arguments corresponds to the type of arguments of the decorated function.
    These arguments are also used as the formal argument of the decorated
    function.  If *func* has the signature ``foo(typing_context, arg0, arg1)``,
    the decorated function will have the signature ``foo(arg0, arg1)``.

    The return values of *func* should be a 2-tuple of expected type signature,
    and a code-generation function that will passed to ``lower_builtin``.
    For unsupported operation, return None.

    Here is an example implementing a ``cast_int_to_byte_ptr`` that cast
    any integer to a byte pointer::

        @intrinsic
        def cast_int_to_byte_ptr(typingctx, src):
            # check for accepted types
            if isinstance(src, types.Integer):
                # create the expected type signature
                result_type = types.CPointer(types.uint8)
                sig = result_type(types.uintp)
                # defines the custom code generation
                def codegen(context, builder, signature, args):
                    # llvm IRBuilder code here
                    [src] = args
                    rtype = signature.return_type
                    llrtype = context.get_value_type(rtype)
                    return builder.inttoptr(src, llrtype)
                return sig, codegen

    Optionally, keyword arguments can be provided to configure the intrinsic; e.g.

        @intrinsic(support_literals=True)
        def example(typingctx, ...):
            ...

    Supported keyword arguments are:

    - support_literals : bool
        Indicates to the type inferencer that the typing logic accepts and can specialize to
        `Const` type.
    """
    # Make inner function for the actual work
    def _intrinsic(func):
        name = getattr(func, '__name__', str(func))
        llc = _Intrinsic(name, func, **kwargs)
        llc._register()
        return llc

    if not kwargs:
        # No option is given
        return _intrinsic(*args)
    else:
        # options are given, create a new callable to recv the
        # definition function
        def wrapper(func):
            return _intrinsic(func)
        return wrapper


def get_cython_function_address(module_name, function_name):
    """
    Get the address of a Cython function.

    Args
    ----
    module_name:
        Name of the Cython module
    function_name:
        Name of the Cython function

    Returns
    -------
    A Python int containing the address of the function

    """
    return _import_cython_function(module_name, function_name)
