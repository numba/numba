from __future__ import print_function, division, absolute_import

import itertools
import weakref

from .six import add_metaclass


# Types are added to a global registry (_typecache) in order to assign
# them unique integer codes for fast matching in _dispatcher.c.
# However, we also want types to be disposable, therefore we ensure
# each type is interned as a weak reference, so that it lives only as
# long as necessary to keep a stable type code.
_typecodes = itertools.count()

def _autoincr():
    n = next(_typecodes)
    # 4 billion types should be enough, right?
    assert n < 2 ** 32, "Limited to 4 billion types"
    return n

_typecache = {}

def _on_type_disposal(wr, _pop=_typecache.pop):
    _pop(wr, None)


class _TypeMetaclass(type):
    """
    A metaclass that will intern instances after they are created.
    This is done by first creating a new instance (including calling
    __init__, which sets up the required attributes for equality
    and hashing), then looking it up in the _typecache registry.
    """

    def __call__(cls, *args, **kwargs):
        """
        Instantiate *cls* (a Type subclass, presumably) and intern it.
        If an interned instance already exists, it is returned, otherwise
        the new instance is returned.
        """
        inst = type.__call__(cls, *args, **kwargs)
        # Try to intern the created instance
        wr = weakref.ref(inst, _on_type_disposal)
        orig = _typecache.get(wr)
        orig = orig and orig()
        if orig is not None:
            return orig
        else:
            inst._code = _autoincr()
            _typecache[wr] = wr
            inst.post_init()
            return inst


@add_metaclass(_TypeMetaclass)
class Type(object):
    """
    The default behavior is to provide equality through `name` attribute.
    Two types are equal if there `name` are equal.
    Subclass can refine this behavior.
    """

    mutable = False

    def __init__(self, name, param=False):
        self.name = name
        self.is_parametric = param

    def post_init(self):
        """
        A method called when the instance is fully initialized and has
        a registered typecode in its _code attribute.  Does nothing by
        default, but can be overriden.
        """

    @property
    def key(self):
        """
        A property used for __eq__, __ne__ and __hash__.  Can be overriden
        in subclasses.
        """
        return self.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.key == other.key

    def __ne__(self, other):
        return not (self == other)

    def coerce(self, typingctx, other):
        """Override this method to implement specialized coercion logic
        for extending unify_pairs().  Only use this if the coercion logic cannot
        be expressed as simple casting rules.
        """
        return NotImplemented

    # User-facing helpers.  These are not part of the core Type API but
    # are provided so that users can write e.g. `numba.boolean(1.5)`
    # (returns True) or `types.int32(types.int32[:])` (returns something
    # usable as a function signature).

    def __call__(self, *args):
        from .typing import signature
        if len(args) == 1 and not isinstance(args[0], Type):
            return self.cast_python_value(args[0])
        return signature(self, # return_type
                         *args)

    def __getitem__(self, args):
        from .types import Array
        ndim, layout = self._determine_array_spec(args)
        return Array(dtype=self, ndim=ndim, layout=layout)

    def _determine_array_spec(self, args):
        if isinstance(args, (tuple, list)):
            ndim = len(args)
            if args[0].step == 1:
                layout = 'F'
            elif args[-1].step == 1:
                layout = 'C'
            else:
                layout = 'A'
        elif isinstance(args, slice):
            ndim = 1
            if args.step == 1:
                layout = 'C'
            else:
                layout = 'A'
        else:
            ndim = 1
            layout = 'A'

        return ndim, layout

    cast_python_value = NotImplemented
    __iter__ = NotImplemented  # ???


class Number(Type):
    """
    Base class for number types.
    """


class Callable(Type):
    """
    Base class for callables.
    """

    def get_call_type(self, context, args, kws):
        """
        Using the typing *context*, resolve the callable's signature for
        the given arguments.  A signature object is returned, or None.
        """
        raise NotImplementedError


class DTypeSpec(Type):
    """
    Base class for types usable as "dtype" arguments to various Numpy APIs
    (e.g. np.empty()).
    """

    @property
    def dtype(self):
        raise NotImplementedError


class IterableType(Type):
    """
    Base class for iterable types.
    Derived classes should implement the *iterator_type* attribute.
    """


class IteratorType(IterableType):
    """
    Base class for all iterator types.
    Derived classes should implement the *yield_type* attribute.
    """

    def __init__(self, name, **kwargs):
        self.iterator_type = self
        super(IteratorType, self).__init__(name, **kwargs)
