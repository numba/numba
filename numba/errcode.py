from __future__ import print_function, absolute_import


class Enum(object):
    def __init__(self, init=0):
        self.value = init

    def get(self):
        v = self.value
        self.value += 1
        return v


enum = Enum().get

FIRST_ERROR = RUNTIME_ERROR = enum()
ASSERTION_ERROR = enum()
INDEX_ERROR = enum()
OUT_OF_BOUND_ERROR = enum()

# Count number of error
ERROR_COUNT = enum() - FIRST_ERROR


def _build_errtable():
    table = {}
    for k, v in globals().items():
        if isinstance(v, int) and v < ERROR_COUNT:
            table[v] = k
    return table


error_names = _build_errtable()

