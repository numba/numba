import enum
import typing as pt
from functools import total_ordering

import numpy as np
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers

from .abstract import Dummy, Hashable, Literal, NumbaTypeInst, Number, Type

_ComparisonOrNotImplemented = pt.Union[bool, "NotImplemented"]


class Boolean(Hashable):

    def cast_python_value(self, value: pt.Any) -> bool:
        return bool(value)


def _parse_bitwith(name: str, *possible_prefixes: str) -> int:
    bitwidth: pt.Optional[int] = None

    for prefix in possible_prefixes:
        if name.startswith(prefix):
            bitwidth = int(name[len(prefix):])

    if bitwidth is None:
        raise ValueError(f"Unable to parse integer with from {name}")

    return bitwidth


def parse_integer_signed(name: str) -> bool:
    signed = name.startswith('int')
    return signed


@total_ordering
class Integer(Number):
    def __init__(
        self,
        name: str,
        bitwidth: pt.Optional[int] = None,
        signed: pt.Optional[bool] = None,
    ):
        super(Integer, self).__init__(name)
        if bitwidth is None:
            bitwidth = _parse_bitwith(name, "int", "uint")
        if signed is None:
            signed = parse_integer_signed(name)
        self.bitwidth = bitwidth
        self.signed = signed

    @classmethod
    def from_bitwidth(cls, bitwidth: int, signed: bool = True):
        name = ('int%d' if signed else 'uint%d') % bitwidth
        return cls(name)

    def cast_python_value(self, value: pt.Any) -> int:
        return getattr(np, self.name)(value)  # type: ignore[no-any-return]

    def __lt__(self, other: pt.Any) -> _ComparisonOrNotImplemented:
        if self.__class__ is not other.__class__:
            return NotImplemented
        if self.signed != other.signed:
            return NotImplemented
        return self.bitwidth < other.bitwidth

    @property
    def maxval(self) -> int:
        """
        The maximum value representable by this type.
        """
        if self.signed:
            return (1 << (self.bitwidth - 1)) - 1
        else:
            return (1 << self.bitwidth) - 1

    @property
    def minval(self) -> int:
        """
        The minimal value representable by this type.
        """
        if self.signed:
            return -(1 << (self.bitwidth - 1))
        else:
            return 0


class IntegerLiteral(Literal, Integer):
    def __init__(self, value: int):
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
        if conv is not None:
            return max(conv, Conversion.promote)


Literal.ctor_map[int] = IntegerLiteral


class BooleanLiteral(Literal, Boolean):

    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[bool]({})'.format(value)
        Boolean.__init__(
            self,
            name=name
            )

    def can_convert_to(self, typingctx, other):
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, Conversion.promote)


Literal.ctor_map[bool] = BooleanLiteral


@total_ordering
class Float(Number):
    def __init__(self, name: str):
        super(Float, self).__init__(name)
        self.bitwidth = _parse_bitwith(name, "float")

    def cast_python_value(self, value: pt.Any) -> float:
        return getattr(np, self.name)(value)  # type: ignore[no-any-return]

    def __lt__(self, other: pt.Any) -> _ComparisonOrNotImplemented:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


@total_ordering
class Complex(Number):
    def __init__(self, name: str, underlying_float: Float):
        super(Complex, self).__init__(name)
        self.underlying_float = underlying_float
        self.bitwidth = _parse_bitwith(name, "complex")

    def cast_python_value(self, value: pt.Any) -> complex:
        return getattr(np, self.name)(value)  # type: ignore[no-any-return]

    def __lt__(self, other: pt.Any) -> _ComparisonOrNotImplemented:
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


class _NPDatetimeBase(Type):
    """
    Common base class for np.datetime64 and np.timedelta64.
    """

    type_name: pt.ClassVar[str]

    def __init__(self, unit: str):
        name = '%s[%s]' % (self.type_name, unit)
        self.unit = unit
        self.unit_code = npdatetime_helpers.DATETIME_UNITS[self.unit]
        super(_NPDatetimeBase, self).__init__(name)

    def __lt__(self, other: pt.Any) -> _ComparisonOrNotImplemented:
        if self.__class__ is not other.__class__:
            return NotImplemented
        # A coarser-grained unit is "smaller", i.e. less precise values
        # can be represented (but the magnitude of representable values is
        # also greater...).
        return self.unit_code < other.unit_code

    def cast_python_value(self, value: pt.Any) -> pt.Any:
        cls = getattr(np, self.type_name)
        if self.unit:
            return cls(value, self.unit)
        else:
            return cls(value)


@total_ordering
class NPTimedelta(_NPDatetimeBase):
    type_name = 'timedelta64'


@total_ordering
class NPDatetime(_NPDatetimeBase):
    type_name = 'datetime64'


class EnumClass(Dummy):
    """
    Type class for Enum classes.
    """
    basename = "Enum class"

    def __init__(self, cls: pt.Type[enum.Enum], dtype: NumbaTypeInst):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = f"{self.basename}<{self.dtype}>({self.instance_class.__name__})"
        super(EnumClass, self).__init__(name)

    @property
    def key(self) -> pt.Any:
        return self.instance_class, self.dtype

    @utils.cached_property
    def member_type(self) -> NumbaTypeInst:
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
    def member_type(self) -> NumbaTypeInst:
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

    def __init__(self, cls: pt.Type[enum.Enum], dtype: NumbaTypeInst):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = f"{self.basename}<{self.dtype}>({self.instance_class.__name__})"
        super(EnumMember, self).__init__(name)

    @property
    def key(self) -> pt.Any:
        return self.instance_class, self.dtype

    @property
    def class_type(self) -> NumbaTypeInst:
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
