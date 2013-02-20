import numba as nb
from numba import *
from numba import typesystem

import numpy as np

GROW = 2

def notimplemented(msg):
    raise NotImplementedError("'%s' method" % msg)

def container_methods(item_type, notimplemented):
    # NOTE: numba will use the global 'notimplemented' function, not the
    # one passed in :(

    @item_type(Py_ssize_t) # TODO: slicing!
    def getitem(self, key):
        if not (0 <= key < self.size):
            # TODO: Implement raise !
            # raise IndexError(key)
            [][key] # tee hee

        return self.buf[key]

    @void(Py_ssize_t, item_type) # TODO: slice assignment!
    def setitem(self, key, value):
        if not (0 <= key < self.size):
            # TODO: Implement raise !
            # raise IndexError(key)
            [][key]

        self.buf[key] = value

    @void(item_type)
    def append(self, value):
        size = self.size
        if size >= self.buf.shape[0]:
            # NOTE: initial bufsize must be greater than zero
            self.buf.resize(int(size * GROW), refcheck=False)

        self.buf[size] = value
        self.size = size + 1

    @void(object_)
    def extend(self, iterable):
        # TODO: something fast for common cases (e.g. typedlist,
        #                                        np.ndarray, etc)
        for obj in iterable:
            self.append(obj)

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

    return locals()