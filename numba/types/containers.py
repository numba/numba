from __future__ import print_function, division, absolute_import

from .abstract import *
from .common import *
from .misc import Undefined
from ..typeconv import Conversion


class Pair(Type):
    """
    A heterogenous pair.
    """

    def __init__(self, first_type, second_type):
        self.first_type = first_type
        self.second_type = second_type
        name = "pair<%s, %s>" % (first_type, second_type)
        super(Pair, self).__init__(name=name)

    @property
    def key(self):
        return self.first_type, self.second_type

    def unify(self, typingctx, other):
        if isinstance(other, Pair):
            first = typingctx.unify_pairs(self.first_type, other.first_type)
            second = typingctx.unify_pairs(self.second_type, other.second_type)
            if first is not None and second is not None:
                return Pair(first, second)


class BaseContainerIterator(SimpleIteratorType):
    """
    Convenience base class for some container iterators.

    Derived classes must implement the *container_class* attribute.
    """

    def __init__(self, container):
        assert isinstance(container, self.container_class), container
        self.container = container
        yield_type = container.dtype
        name = 'iter(%s)' % container
        super(BaseContainerIterator, self).__init__(name, yield_type)

    def unify(self, typingctx, other):
        cls = type(self)
        if isinstance(other, cls):
            container = typingctx.unify_pairs(self.container, other.container)
            if container is not None:
                return cls(container)

    @property
    def key(self):
        return self.container


class BaseContainerPayload(Type):
    """
    Convenience base class for some container payloads.

    Derived classes must implement the *container_class* attribute.
    """

    def __init__(self, container):
        assert isinstance(container, self.container_class)
        self.container = container
        name = 'payload(%s)' % container
        super(BaseContainerPayload, self).__init__(name)

    @property
    def key(self):
        return self.container


class Bytes(Buffer):
    """
    Type class for Python 3.x bytes objects.
    """
    mutable = False
    # Actually true but doesn't matter since bytes is immutable
    slice_is_copy = False


class ByteArray(Buffer):
    """
    Type class for bytearray objects.
    """
    slice_is_copy = True


class PyArray(Buffer):
    """
    Type class for array.array objects.
    """
    slice_is_copy = True


class MemoryView(Buffer):
    """
    Type class for memoryview objects.
    """


class BaseTuple(ConstSized, Hashable):
    """
    The base class for all tuple types (with a known size).
    """

    @classmethod
    def from_types(cls, tys, pyclass=None):
        """
        Instantiate the right tuple type for the given element types.
        """
        homogenous = False
        if tys:
            first = tys[0]
            for ty in tys[1:]:
                if ty != first:
                    break
            else:
                homogenous = True

        if pyclass is not None and pyclass is not tuple:
            # A subclass => is it a namedtuple?
            assert issubclass(pyclass, tuple)
            if hasattr(pyclass, "_asdict"):
                if homogenous:
                    return NamedUniTuple(first, len(tys), pyclass)
                else:
                    return NamedTuple(tys, pyclass)
        if homogenous:
            return UniTuple(first, len(tys))
        else:
            return Tuple(tys)


class BaseAnonymousTuple(BaseTuple):
    """
    Mixin for non-named tuples.
    """

    def can_convert_to(self, typingctx, other):
        """
        Convert this tuple to another one.  Note named tuples are rejected.
        """
        if not isinstance(other, BaseAnonymousTuple):
            return
        if len(self) != len(other):
            return
        if len(self) == 0:
            return Conversion.safe
        if isinstance(other, BaseTuple):
            kinds = [typingctx.can_convert(ta, tb)
                     for ta, tb in zip(self, other)]
            if any(kind is None for kind in kinds):
                return
            return max(kinds)


class _HomogenousTuple(Sequence, BaseTuple):

    @property
    def iterator_type(self):
        return UniTupleIter(self)

    def getitem(self, ind):
        return self.dtype, intp

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.dtype

    def __iter__(self):
        return iter([self.dtype] * self.count)

    def __len__(self):
        return self.count

    @property
    def types(self):
        return (self.dtype,) * self.count


class UniTuple(BaseAnonymousTuple, _HomogenousTuple, Sequence):
    """
    Type class for homogenous tuples.
    """

    def __init__(self, dtype, count):
        self.dtype = dtype
        self.count = count
        name = "(%s x %d)" % (dtype, count)
        super(UniTuple, self).__init__(name)

    @property
    def key(self):
        return self.dtype, self.count

    def unify(self, typingctx, other):
        """
        Unify UniTuples with their dtype
        """
        if isinstance(other, UniTuple) and len(self) == len(other):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            if dtype is not None:
                return UniTuple(dtype=dtype, count=self.count)


class UniTupleIter(BaseContainerIterator):
    """
    Type class for homogenous tuple iterators.
    """
    container_class = _HomogenousTuple


class _HeterogenousTuple(BaseTuple):

    def __getitem__(self, i):
        """
        Return element at position i
        """
        return self.types[i]

    def __len__(self):
        # Beware: this makes Tuple(()) false-ish
        return len(self.types)

    def __iter__(self):
        return iter(self.types)


class Tuple(BaseAnonymousTuple, _HeterogenousTuple):

    def __new__(cls, types):
        if types and all(t == types[0] for t in types[1:]):
            return UniTuple(dtype=types[0], count=len(types))
        else:
            return object.__new__(Tuple)

    def __init__(self, types):
        self.types = tuple(types)
        self.count = len(self.types)
        name = "(%s)" % ', '.join(str(i) for i in self.types)
        super(Tuple, self).__init__(name)

    @property
    def key(self):
        return self.types

    def unify(self, typingctx, other):
        """
        Unify elements of Tuples/UniTuples
        """
        # Other is UniTuple or Tuple
        if isinstance(other, BaseTuple) and len(self) == len(other):
            unified = [typingctx.unify_pairs(ta, tb)
                       for ta, tb in zip(self, other)]

            if all(t is not None for t in unified):
                return Tuple(unified)


class BaseNamedTuple(BaseTuple):
    pass


class NamedUniTuple(_HomogenousTuple, BaseNamedTuple):

    def __init__(self, dtype, count, cls):
        self.dtype = dtype
        self.count = count
        self.fields = tuple(cls._fields)
        self.instance_class = cls
        name = "%s(%s x %d)" % (cls.__name__, dtype, count)
        super(NamedUniTuple, self).__init__(name)

    @property
    def iterator_type(self):
        return UniTupleIter(self)

    @property
    def key(self):
        return self.instance_class, self.dtype, self.count


class NamedTuple(_HeterogenousTuple, BaseNamedTuple):

    def __init__(self, types, cls):
        self.types = tuple(types)
        self.count = len(self.types)
        self.fields = tuple(cls._fields)
        self.instance_class = cls
        name = "%s(%s)" % (cls.__name__, ', '.join(str(i) for i in self.types))
        super(NamedTuple, self).__init__(name)

    @property
    def key(self):
        return self.instance_class, self.types


class List(MutableSequence):
    """
    Type class for (arbitrary-sized) homogenous lists.
    """
    mutable = True

    def __init__(self, dtype, reflected=False):
        self.dtype = dtype
        self.reflected = reflected
        cls_name = "reflected list" if reflected else "list"
        name = "%s(%s)" % (cls_name, self.dtype)
        super(List, self).__init__(name=name)

    def copy(self, dtype=None, reflected=None):
        if dtype is None:
            dtype = self.dtype
        if reflected is None:
            reflected = self.reflected
        return List(dtype, reflected)

    def unify(self, typingctx, other):
        if isinstance(other, List):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            reflected = self.reflected or other.reflected
            if dtype is not None:
                return List(dtype, reflected)

    @property
    def key(self):
        return self.dtype, self.reflected

    @property
    def iterator_type(self):
        return ListIter(self)

    def is_precise(self):
        return self.dtype.is_precise()


class ListIter(BaseContainerIterator):
    """
    Type class for list iterators.
    """
    container_class = List


class ListPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a list.
    """
    container_class = List


class Set(Container):
    """
    Type class for homogenous sets.
    """
    mutable = True

    def __init__(self, dtype, reflected=False):
        assert isinstance(dtype, (Hashable, Undefined))
        self.dtype = dtype
        self.reflected = reflected
        cls_name = "reflected set" if reflected else "set"
        name = "%s(%s)" % (cls_name, self.dtype)
        super(Set, self).__init__(name=name)

    @property
    def key(self):
        return self.dtype, self.reflected

    @property
    def iterator_type(self):
        return SetIter(self)

    def is_precise(self):
        return self.dtype.is_precise()

    def copy(self, dtype=None, reflected=None):
        if dtype is None:
            dtype = self.dtype
        if reflected is None:
            reflected = self.reflected
        return Set(dtype, reflected)

    def unify(self, typingctx, other):
        if isinstance(other, Set):
            dtype = typingctx.unify_pairs(self.dtype, other.dtype)
            reflected = self.reflected or other.reflected
            if dtype is not None:
                return Set(dtype, reflected)


class SetIter(BaseContainerIterator):
    """
    Type class for set iterators.
    """
    container_class = Set


class SetPayload(BaseContainerPayload):
    """
    Internal type class for the dynamically-allocated payload of a set.
    """
    container_class = Set


class SetEntry(Type):
    """
    Internal type class for the entries of a Set's hash table.
    """
    def __init__(self, set_type):
        self.set_type = set_type
        name = 'entry(%s)' % set_type
        super(SetEntry, self).__init__(name)

    @property
    def key(self):
        return self.set_type
