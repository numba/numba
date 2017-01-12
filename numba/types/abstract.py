from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
import itertools
import weakref

import numpy as np

from ..six import add_metaclass
from ..utils import cached_property


# Types are added to a global registry (_typecache) in order to assign
# them unique integer codes for fast matching in _dispatcher.c.
# However, we also want types to be disposable, therefore we ensure
# each type is interned as a weak reference, so that it lives only as
# long as necessary to keep a stable type code.
# NOTE: some types can still be made immortal elsewhere (for example
# in _dispatcher.c's internal caches).
_typecodes = itertools.count()

def _autoincr():
    n = next(_typecodes)
    # 4 billion types should be enough, right?
    assert n < 2 ** 32, "Limited to 4 billion types"
    return n

_typecache = {}

def _on_type_disposal(wr, _pop=_typecache.pop):
    _pop(wr, None)


class _TypeMetaclass(ABCMeta):
    """
    A metaclass that will intern instances after they are created.
    This is done by first creating a new instance (including calling
    __init__, which sets up the required attributes for equality
    and hashing), then looking it up in the _typecache registry.
    """

    def _intern(cls, inst):
        # Try to intern the created instance
        wr = weakref.ref(inst, _on_type_disposal)
        orig = _typecache.get(wr)
        orig = orig and orig()
        if orig is not None:
            return orig
        else:
            inst._code = _autoincr()
            _typecache[wr] = wr
            return inst

    def __call__(cls, *args, **kwargs):
        """
        Instantiate *cls* (a Type subclass, presumably) and intern it.
        If an interned instance already exists, it is returned, otherwise
        the new instance is returned.
        """
        inst = type.__call__(cls, *args, **kwargs)
        return cls._intern(inst)


def _type_reconstructor(reconstructor, reconstructor_args, state):
    """
    Rebuild function for unpickling types.
    """
    obj = reconstructor(*reconstructor_args)
    if state:
        obj.__dict__.update(state)
    return type(obj)._intern(obj)


@add_metaclass(_TypeMetaclass)
class Type(object):
    """
    The base class for all Numba types.
    It is essential that proper equality comparison is implemented.  The
    default implementation uses the "key" property (overridable in subclasses)
    for both comparison and hashing, to ensure sane behaviour.
    """

    mutable = False

    def __init__(self, name):
        self.name = name

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

    def __reduce__(self):
        reconstructor, args, state = super(Type, self).__reduce__()
        return (_type_reconstructor, (reconstructor, args, state))

    def unify(self, typingctx, other):
        """
        Try to unify this type with the *other*.  A third type must
        be returned, or None if unification is not possible.
        Only override this if the coercion logic cannot be expressed
        as simple casting rules.
        """
        return None

    def can_convert_to(self, typingctx, other):
        """
        Check whether this type can be converted to the *other*.
        If successful, must return a string describing the conversion, e.g.
        "exact", "promote", "unsafe", "safe"; otherwise None is returned.
        """
        return None

    def can_convert_from(self, typingctx, other):
        """
        Similar to *can_convert_to*, but in reverse.  Only needed if
        the type provides conversion from other types.
        """
        return None

    def is_precise(self):
        """
        Whether this type is precise, i.e. can be part of a successful
        type inference.  Default implementation returns True.
        """
        return True

    def augment(self, other):
        """
        Augment this type with the *other*.  Return the augmented type,
        or None if not supported.
        """
        return None

    # User-facing helpers.  These are not part of the core Type API but
    # are provided so that users can write e.g. `numba.boolean(1.5)`
    # (returns True) or `types.int32(types.int32[:])` (returns something
    # usable as a function signature).

    def __call__(self, *args):
        from ..typing import signature
        if len(args) == 1 and not isinstance(args[0], Type):
            return self.cast_python_value(args[0])
        return signature(self, # return_type
                         *args)

    def __getitem__(self, args):
        """
        Return an array of this type.
        """
        from . import Array
        ndim, layout = self._determine_array_spec(args)
        return Array(dtype=self, ndim=ndim, layout=layout)

    def _determine_array_spec(self, args):
        # XXX non-contiguous by default, even for 1d arrays,
        # doesn't sound very intuitive
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

    def cast_python_value(self, args):
        raise NotImplementedError


# XXX we should distinguish between Dummy (no meaningful
# representation, e.g. None or a builtin function) and Opaque (has a
# meaningful representation, e.g. ExternalFunctionPointer)

class Dummy(Type):
    """
    Base class for types that do not really have a representation and are
    compatible with a void*.
    """


class Hashable(Type):
    """
    Base class for hashable types.
    """


class Number(Hashable):
    """
    Base class for number types.
    """

    def unify(self, typingctx, other):
        """
        Unify the two number types using Numpy's rules.
        """
        from .. import numpy_support
        if isinstance(other, Number):
            # XXX: this can produce unsafe conversions,
            # e.g. would unify {int64, uint64} to float64
            a = numpy_support.as_dtype(self)
            b = numpy_support.as_dtype(other)
            sel = np.promote_types(a, b)
            return numpy_support.from_dtype(sel)


class Callable(Type):
    """
    Base class for callables.
    """

    @abstractmethod
    def get_call_type(self, context, args, kws):
        """
        Using the typing *context*, resolve the callable's signature for
        the given arguments.  A signature object is returned, or None.
        """

    @abstractmethod
    def get_call_signatures(self):
        """
        Returns a tuple of (list of signatures, parameterized)
        """


class DTypeSpec(Type):
    """
    Base class for types usable as "dtype" arguments to various Numpy APIs
    (e.g. np.empty()).
    """

    @abstractproperty
    def dtype(self):
        """
        The actual dtype denoted by this dtype spec (a Type instance).
        """


class IterableType(Type):
    """
    Base class for iterable types.
    """

    @abstractproperty
    def iterator_type(self):
        """
        The iterator type obtained when calling iter() (explicitly or implicitly).
        """


class Sized(Type):
    """
    Base class for objects that support len()
    """


class ConstSized(Sized):
    """
    For types that have a constant size
    """
    @abstractmethod
    def __len__(self):
        pass


class IteratorType(IterableType):
    """
    Base class for all iterator types.
    Derived classes should implement the *yield_type* attribute.
    """

    def __init__(self, name, **kwargs):
        super(IteratorType, self).__init__(name, **kwargs)

    @abstractproperty
    def yield_type(self):
        """
        The type of values yielded by the iterator.
        """

    # This is a property to avoid recursivity (for pickling)

    @property
    def iterator_type(self):
        return self


class Container(Sized, IterableType):
    """
    Base class for container types.
    """


class Sequence(Container):
    """
    Base class for 1d sequence types.  Instances should have the *dtype*
    attribute.
    """


class MutableSequence(Sequence):
    """
    Base class for 1d mutable sequence types.  Instances should have the
    *dtype* attribute.
    """


class ArrayCompatible(Type):
    """
    Type class for Numpy array-compatible objects (typically, objects
    exposing an __array__ method).
    Derived classes should implement the *as_array* attribute.
    """
    # If overriden by a subclass, it should also implement typing
    # for '__array_wrap__' with arguments (input, formal result).
    array_priority = 0.0

    @abstractproperty
    def as_array(self):
        """
        The equivalent array type, for operations supporting array-compatible
        objects (such as ufuncs).
        """

    # For compatibility with types.Array

    @cached_property
    def ndim(self):
        return self.as_array.ndim

    @cached_property
    def layout(self):
        return self.as_array.layout

    @cached_property
    def dtype(self):
        return self.as_array.dtype
