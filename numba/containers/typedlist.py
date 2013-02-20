import numba as nb
from numba import *
from numba import typesystem

import numpy as np

INITIAL_BUFSIZE = 10
SHRINK = 1.5
GROW = 2

def notimplemented(msg):
    raise NotImplementedError("'%s' method of type 'typedlist'" % msg)

_list_cache = {}

def typedlist(item_type, iterable=None, _list_cache=_list_cache):
    if item_type in _list_cache:
        return _list_cache[item_type](item_type, item_type.get_dtype(), iterable)

    item_type_t = typesystem.CastType(item_type)
    dtype_t = typesystem.NumpyDtypeType(item_type)

    @jit(warn=False)
    class typedlist(object):
        @void(item_type_t, dtype_t, object_)
        def __init__(self, item_type, dtype, iterable):
            # self.item_type = item_type
            item_type
            self.dtype = dtype
            self.size = 0

            # TODO: Use length hint of iterable for initial buffer size
            self.buf = np.empty(INITIAL_BUFSIZE, dtype=dtype)

            # TODO: implement 'is'/'is not'
            if iterable != None:
                self.extend(iterable)

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
                self.buf.resize(int(size * GROW), refcheck=False)

            self.buf[size] = value
            self.size = size + 1

        @void(object_)
        def extend(self, iterable):
            # TODO: something fast for common cases (e.g. typedlist,
            #                                        np.ndarray, etc)
            for obj in iterable:
                self.append(obj)

        @item_type()
        def pop(self):
            # TODO: Optional argument 'index'
            size = self.size - 1
            item = self.buf[size]
            self.size = size

            if INITIAL_BUFSIZE < size < self.buf.shape[0] / 2:
                self.buf.resize(int(SHRINK * size), refcheck=False)

            return item

        @Py_ssize_t(item_type)
        def index(self, value):
            self, value
            notimplemented("index")
            # TODO: comparison of complex numbers (#121)
            # buf = self.buf
            # for i in range(self.size):
            #     if buf[i] == value:
            #         return i

            # [].index(value) # raise ValueError

        @Py_ssize_t(item_type)
        def count(self, value):
            self, value
            notimplemented("count")
            # TODO: comparison of complex numbers (#121)
            # count = 0
            # buf = self.buf
            # for i in range(self.size):
            #     # TODO: promotion of (bool_, int_)
            #     count += Py_ssize_t(buf[i] == value)

            # return count

        @void(Py_ssize_t, item_type)
        def insert(self, index, value):
            self, index, value # TODO: implemented specifying warn=False
            notimplemented("insert")

        @void(item_type)
        def remove(self, value):
            self, value
            notimplemented("remove")

        @void()
        def reverse(self):
            buf = self.buf
            size = self.size - 1
            for i in range(self.size / 2):
                tmp = buf[i]
                buf[i] = buf[size - i]
                buf[size - i] = tmp

        @void()
        def sort(self):
            # TODO: optional arguments cmp, key, reverse
            self
            notimplemented("sort")

        @Py_ssize_t()
        def __len__(self):
            return self.size

        @nb.c_string_type()
        def __repr__(self):
            buf = ", ".join([str(self.buf[i]) for i in range(self.size)])
            return "[" + buf + "]"

    _list_cache[item_type] = typedlist
    return typedlist(item_type, item_type.get_dtype(), iterable)


if __name__ == "__main__":
    result = typedlist(int_)
    for i in range(50):
        result.append(i)

    print result.buf
    print result
    print result[100]