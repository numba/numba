from __future__ import print_function, division, absolute_import

import enum

import numpy

from .abstract import *
from .. import npdatetime, utils


class Boolean(Hashable):

    def cast_python_value(self, value):
        return bool(value)


@utils.total_ordering
class Integer(Number):
    def __init__(self, *args, **kws):
        super(Integer, self).__init__(*args, **kws)
        # Determine bitwidth
        for prefix in ('int', 'uint'):
            if self.name.startswith(prefix):
                bitwidth = int(self.name[len(prefix):])
        self.bitwidth = bitwidth
        self.signed = self.name.startswith('int')

    @classmethod
    def from_bitwidth(cls, bitwidth, signed=True):
        name = ('int%d' if signed else 'uint%d') % bitwidth
        return cls(name)

    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        if self.signed != other.signed:
            return NotImplemented
        return self.bitwidth < other.bitwidth


@utils.total_ordering
class Float(Number):
    def __init__(self, *args, **kws):
        super(Float, self).__init__(*args, **kws)
        # Determine bitwidth
        assert self.name.startswith('float')
        bitwidth = int(self.name[5:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return getattr(numpy, self.name)(value)

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
        return getattr(numpy, self.name)(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


class _NPDatetimeBase(Type):
    """
    Common base class for numpy.datetime64 and numpy.timedelta64.
    """

    def __init__(self, unit, *args, **kws):
        name = '%s(%s)' % (self.type_name, unit)
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
        cls = getattr(numpy, self.type_name)
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
    Type class for enum classes.
    """

    def __init__(self, cls, dtype):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = "enum class<%s>(%s)" % (self.dtype, self.instance_class.__name__)
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


class EnumMember(Type):
    """
    Type class for enum members.
    """

    def __init__(self, cls, dtype):
        assert isinstance(cls, type)
        assert isinstance(dtype, Type)
        self.instance_class = cls
        self.dtype = dtype
        name = "enum<%s>(%s)" % (self.dtype, self.instance_class.__name__)
        super(EnumMember, self).__init__(name)

    @property
    def key(self):
        return self.instance_class, self.dtype

    @property
    def class_type(self):
        """
        The type of this member's class.
        """
        return EnumClass(self.instance_class, self.dtype)
