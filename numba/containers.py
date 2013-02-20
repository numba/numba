import numba as nb
from numba import *
from numba import typesystem

import numpy as np

INITIAL_BUFSIZE = 10
SHRINK = 1.5
GROW = 2

_list_cache = {}

def typedlist(item_type, _list_cache=_list_cache):
    if item_type in _list_cache:
        return _list_cache[item_type](item_type, item_type.get_dtype())

    item_type_t = typesystem.CastType(item_type)
    dtype_t = typesystem.NumpyDtypeType(item_type)

    @jit
    class typedlist(object):
        @void(item_type_t, dtype_t)
        def __init__(self, item_type, dtype):
            # self.item_type = item_type
            item_type
            self.dtype = dtype
            self.size = 0
            self.buf = np.empty(INITIAL_BUFSIZE, dtype=dtype)

        # TODO: Jit __getitem__/__setitem__ of numba extension types

        @item_type(Py_ssize_t)
        def __getitem__(self, key):
            if not (0 <= key < self.size):
                # TODO: Implement raise !
                # raise IndexError(key)
                [][key] # tee hee

            return self.buf[key]

        @void(Py_ssize_t, item_type)
        def __setitem__(self, key, value):
            if not (0 <= key < self.size):
                # TODO: Implement raise !
                # raise IndexError(key)
                [][key]

            self.buf[key] = value

        @void(item_type)
        def append(self, value):
            size = self.size
            if size >= self.buf.shape[0]:
                self.buf.resize(int(size * 2))

            self.buf[size] = value
            self.size = size + 1

        @item_type()
        def pop(self):
            # TODO: Optional argument 'index'
            size = self.size - 1
            item = self.buf[size]
            self.size = size

            if INITIAL_BUFSIZE < size < self.buf.shape[0] / 2:
                self.buf.resize(int(SHRINK * size))

            return item

        @Py_ssize_t()
        def __len__(self):
            return self.size

        @nb.c_string_type()
        def __repr__(self):
            buf = ", ".join([str(self.buf[i]) for i in range(self.size)])
            return "[" + buf + "]"

    _list_cache[item_type] = typedlist
    return typedlist(item_type, item_type.get_dtype())


if __name__ == "__main__":
    result = typedlist(int_)
    for i in range(50):
        result.append(i)

    print result.buf
    print result
    print result[100]