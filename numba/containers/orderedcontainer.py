# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numba as nb
from numba import *
from numba import nodes
from numba import typesystem
from numba.typesystem import get_type

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
        # TODO: comparison of complex numbers (#121)
        buf = self.buf
        for i in range(self.size):
             if buf[i] == value:
                 return i

        [].index(value) # raise ValueError

    @Py_ssize_t(item_type)
    def count(self, value):
        # TODO: comparison of complex numbers (#121)
        count = 0
        buf = self.buf
        for i in range(self.size):
             # TODO: promotion of (bool_, int_)
             if buf[i] == value:
                 count += 1

        return count

    return locals()

#-----------------------------------------------------------------------
# Infer types for typed containers (typedlist, typedtuple)
#-----------------------------------------------------------------------

def typedcontainer_infer(compile_typedcontainer, type_node, iterable_node):
    """
    Type inferer for typed containers, register with numba.register_inferer().

    :param compile_typedcontainer: item_type -> typed container extension class
    :param type_node: type parameter to typed container constructor
    :param iterable_node: value parameter to typed container constructor (optional)
    """
    assert type_node is not None

    type = get_type(type_node)
    if type.is_cast:
        elem_type = type.dst_type

        # Pre-compile typed list implementation
        typedcontainer_ctor = compile_typedcontainer(elem_type)

        # Inject the typedlist directly to avoid runtime implementation lookup
        iterable_node = iterable_node or nodes.const(None, object_)
        result = nodes.call_pyfunc(typedcontainer_ctor, (iterable_node,))
        return nodes.CoercionNode(result, typedcontainer_ctor.exttype)

    return object_
