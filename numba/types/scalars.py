from __future__ import print_function, division, absolute_import

import enum

import numpy as np

from .abstract import *
from .. import npdatetime, utils
from ..typeconv import Conversion


class Boolean(Hashable):

    def cast_python_value(self, value):
        return bool(value)


def parse_integer_bitwidth(name):
    for prefix in ('int', 'uint'):
        if name.startswith(prefix):
            bitwidth = int(name[len(prefix):])
    return bitwidth


def parse_integer_signed(name):
    signed = name.startswith('int')
    return signed


@utils.total_ordering
class Integer(Number):
    def __init__(self, name, bitwidth=None, signed=None):
        super(Integer, self).__init__(name)
        if bitwidth is None:
            bitwidth = parse_integer_bitwidth(name)
        if signed is None:
            signed = parse_integer_signed(name)
        self.bitwidth = bitwidth
        self.signed = signed

    @classmethod
    def from_bitwidth(cls, bitwidth, signed=True):
        name = ('int%d' if signed else 'uint%d') % bitwidth
        return cls(name)

    def cast_python_value(self, value):
        return getattr(np, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        if self.signed != other.signed:
            return NotImplemented
        return self.bitwidth < other.bitwidth

    @property
    def maxval(self):
        """
        The maximum value representable by this type.
        """
        if self.signed:
            return (1 << (self.bitwidth - 1)) - 1
        else:
            return (1 << self.bitwidth) - 1

    @property
    def minval(self):
        """
        The minimal value representable by this type.
        """
        if self.signed:
            return -(1 << (self.bitwidth - 1))
        else:
            return 0


class IntegerLiteral(Literal, Integer):
    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[int]({})'.format(value)
        basetype = self.literal_type
        Integer.__init__(
            self,
            name=name,
            bitwidth=basetype.bitwidth,
            signed=basetype.signed,
            )

    def can_convert_to(self, typingctx, other):
        conv = typingctx.can_convert(self.literal_type, other)
        if conv == Conversion.unsafe:
            try:
                if (
                    isinstance(other, Integer)
                    and other.minval <= self.literal_value <= other.maxval
                ):
                    # TODO: 4494 refine unsafe casts if we can ensure we are within other's range (1 -> uint, ...)
                    conv = Conversion.safe
                elif isinstance(other, Number) and utils.bit_length(self.literal_value) < 32:
                    # TODO: 4494 try int32 -> other if we value is small enough
                    from ..types import int32
                    conv = typingctx.can_convert(int32, other)
            except:
                # should only happen if x <= y <= z goes wrong above
                pass

        if conv is not None:
            return max(conv, Conversion.promote)

    def unify(self, typingctx, other):
        """
        Unify the two number types using Numpy's rules.
        """
        # TODO: 4494 special case (val in range of int32) -> float32
        from ..types import float32
        bitwidth = (utils.bit_length(self.literal_value) // 32 + 1) * 32
        if other == float32 and bitwidth <= 32:
            return float32
        elif isinstance(other, Integer) and not isinstance(other, Literal):
            min_unliteral_type = other.from_bitwidth(bitwidth, other.signed)
            return typingctx.unify_pairs(min_unliteral_type, other)
        from .. import numpy_support
        if isinstance(other, Number):
            # XXX: this can produce unsafe conversions,
            # e.g. would unify {int64, uint64} to float64
            a = numpy_support.as_dtype(self)
            b = numpy_support.as_dtype(other)
            sel = np.promote_types(a, b)
            return numpy_support.from_dtype(sel)


Literal.ctor_map[int] = IntegerLiteral


@utils.total_ordering
class Float(Number):
    def __init__(self, *args, **kws):
        super(Float, self).__init__(*args, **kws)
        # Determine bitwidth
        assert self.name.startswith('float')
        bitwidth = int(self.name[5:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(np, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


@utils.total_ordering
class Complex(Number):
    def __init__(self, name, underlying_float, **kwargs):
        super(Complex, self).__init__(name, **kwargs)
        self.underlying_float = underlying_float
        # Determine bitwidth
        assert self.name.startswith('complex')
        bitwidth = int(self.name[7:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(np, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


class _NPDatetimeBase(Type):
    """
    Common base class for np.datetime64 and np.timedelta64.
    """

    def __init__(self, unit, *args, **kws):
        name = '%s[%s]' % (self.type_name, unit)
        self.unit = unit
        self.unit_code = npdatetime.DATETIME_UNITS[self.unit]
        super(_NPDatetimeBase, self).__init__(name, *args, **kws)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        # A coarser-grained unit is "smaller", i.e. less precise values
        # can be represented (but the magnitude of representable values is
        # also greater...).
        return self.unit_code < other.unit_code

    def cast_python_value(self, value):
        cls = getattr(np, self.type_name)
        if self.unit:
            return cls(value, self.unit)
        else:
            return cls(value)


@utils.total_ordering
class NPTimedelta(_NPDatetimeBase):
    type_name = 'timedelta64'

@utils.total_ordering
class NPDatetime(_NPDatetimeBase):
    type_name = 'datetime64'


class EnumClass(Dummy):
    """
    Type class for Enum classes.
    """
    basename = "Enum class"

    def __init__(self, cls, dtype):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = "%s<%s>(%s)" % (self.basename, self.dtype, self.instance_class.__name__)
        super(EnumClass, self).__init__(name)

    @property
    def key(self):
        return self.instance_class, self.dtype

    @utils.cached_property
    def member_type(self):
        """
        The type of this class' members.
        """
        return EnumMember(self.instance_class, self.dtype)


class IntEnumClass(EnumClass):
    """
    Type class for IntEnum classes.
    """
    basename = "IntEnum class"

    @utils.cached_property
    def member_type(self):
        """
        The type of this class' members.
        """
        return IntEnumMember(self.instance_class, self.dtype)


class EnumMember(Type):
    """
    Type class for Enum members.
    """
    basename = "Enum"
    class_type_class = EnumClass

    def __init__(self, cls, dtype):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = "%s<%s>(%s)" % (self.basename, self.dtype, self.instance_class.__name__)
        super(EnumMember, self).__init__(name)

    @property
    def key(self):
        return self.instance_class, self.dtype

    @property
    def class_type(self):
        """
        The type of this member's class.
        """
        return self.class_type_class(self.instance_class, self.dtype)


class IntEnumMember(EnumMember):
    """
    Type class for IntEnum members.
    """
    basename = "IntEnum"
    class_type_class = IntEnumClass

    def can_convert_to(self, typingctx, other):
        """
        Convert IntEnum members to plain integers.
        """
        if issubclass(self.instance_class, enum.IntEnum):
            conv = typingctx.can_convert(self.dtype, other)
            return max(conv, Conversion.safe)
