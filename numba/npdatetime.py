
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
    return src == dest
