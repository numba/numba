
import numpy as np


DATETIME_UNITS = {
    'Y': 0,   # Years
    'M': 1,   # Months
    'W': 2,   # Weeks
    # Yes, there's a gap here
    'D': 4,   # Days
    'h': 5,   # Hours
    'm': 6,   # Minutes
    's': 7,   # Seconds
    'ms': 8,  # Milliseconds
    'us': 9,  # Microseconds
    'ns': 10, # Nanoseconds
    'ps': 11, # Picoseconds
    'fs': 12, # Femtoseconds
    'as': 13, # Attoseconds
    '': 14,   # "generic", i.e. unit-less
}

# Numpy's special "Not a Time" value (should be equal to -2**63)
NAT = np.timedelta64('nat').astype(int)


def can_cast_timedelta_units(src, dest):
    # Mimick numpy's "safe" casting
    # `dest` must be more precise than `src` and they must be compatible
    # for conversion.
    src = DATETIME_UNITS[src]
    dest = DATETIME_UNITS[dest]
    if src == dest:
        return True
    if src > dest:
        return False
    if src == 14 or dest == 14:
        # unit-less timedelta64 is not compatible with anything else
        return False
    if src <= 1 and dest > 1:
        # Cannot convert between months or years and other units
        return False
    return True


# Exact conversion factors from one unit to the immediately more precise one
_factors = {
    0: (1, 12),   # Years -> Months
    2: (4, 7),    # Weeks -> Days
    4: (5, 24),   # Days -> Hours
    5: (6, 60),   # Hours -> Minutes
    6: (7, 60),   # Minutes -> Seconds
    7: (8, 1000),
    8: (9, 1000),
    9: (10, 1000),
    10: (11, 1000),
    11: (12, 1000),
    12: (13, 1000),
}

def _get_conversion_multiplier(big_unit_code, small_unit_code):
    """
    Return an integer multiplier allowing to convert from *big_unit_code*
    to *small_unit_code*.
    None is returned if the conversion is not possible through a
    simple integer multiplication.
    """
    # Mimicks get_datetime_units_factor() in numpy's datetime.c
    c = big_unit_code
    factor = 1
    while c < small_unit_code:
        try:
            c, mult = _factors[c]
        except KeyError:
            # No possible conversion
            return None
        factor *= mult
    if c == small_unit_code:
        return factor
    else:
        return None

def get_timedelta_conversion_factor(big_unit, small_unit):
    """
    Return an integer multiplier allowing to convert from timedeltas
    of *big_unit* to *small_unit*.
    """
    return _get_conversion_multiplier(DATETIME_UNITS[big_unit],
                                      DATETIME_UNITS[small_unit])

