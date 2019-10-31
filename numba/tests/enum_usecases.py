from __future__ import print_function

from enum import Enum, IntEnum


class Color(Enum):
    red = 1
    green = 2
    blue = 3


class Shake(Enum):
    vanilla = 7
    chocolate = 4
    cookies = 9
    # Same as Color.blue
    mint = 3


class Planet(Enum):
    MERCURY = (3.303e+23, 2.4397e6)
    VENUS   = (4.869e+24, 6.0518e6)
    EARTH   = (5.976e+24, 6.37814e6)
    MARS    = (6.421e+23, 3.3972e6)
    JUPITER = (1.9e+27,   7.1492e7)
    SATURN  = (5.688e+26, 6.0268e7)
    URANUS  = (8.686e+25, 2.5559e7)
    NEPTUNE = (1.024e+26, 2.4746e7)


class HeterogeneousEnum(Enum):
    red = 1.0
    green = 2.0
    blue = 3j


class Shape(IntEnum):
    # Same as Color.green
    circle = 2
    # Same as RequestError.internal_error
    square = 500


class RequestError(IntEnum):
    dummy = 2
    not_found = 404
    internal_error = 500

