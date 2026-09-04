import datetime as dt
from typing import ClassVar, Literal, TypeAlias

import numpy as np
from typing_extensions import Self, override

from numba.core import types

###

_Unit: TypeAlias = Literal[
    "Y",  # years
    "M",  # months
    "W",  # weeks
    "D",  # days
    "h",  # hours
    "m",  # minutes
    "s",  # seconds
    "ms",  # milliseconds
    "us",  # microseconds
    "ns",  # nanoseconds
    "ps",  # picoseconds
    "fs",  # femtoseconds
    "as",  # attoseconds
    "",  # generic / unit-less
]

###
# the following classes are originally defined in `datetime.py`, but stubbed here to
# avoid having to stub the many undocumented internal helper classes in `datetime.py`.

class _NPDatetimeBase(types.Type):
    type_name: ClassVar[str]

    unit: _Unit
    unit_code: int

    def __init__(self, unit: _Unit) -> None: ...
    def __lt__(self, other: Self, /) -> bool: ...

# we can't use `@total_ordering` because of a bug in stubtest (mypy)

class NPTimedelta(_NPDatetimeBase):
    type_name: ClassVar[str] = "timedelta64"

    @override
    def cast_python_value(
        self,
        value: int | str | dt.timedelta | np.timedelta64 | None,
    ) -> np.timedelta64: ...

    # dynamic methods added by `functools.total_ordering`
    def __le__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...

class NPDatetime(_NPDatetimeBase):
    type_name: ClassVar[str] = "datetime64"

    @override
    def cast_python_value(
        self,
        value: int | str | dt.date | np.datetime64 | None,
    ) -> np.datetime64: ...

    # dynamic methods added by `functools.total_ordering`
    def __le__(self, other: Self, /) -> bool: ...
    def __gt__(self, other: Self, /) -> bool: ...
    def __ge__(self, other: Self, /) -> bool: ...
