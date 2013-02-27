import numba as nb
from numba import *
from numba import nodes
from numba import typesystem
from numba.typesystem import get_type
from numba.containers import orderedcontainer
from numba.type_inference.module_type_inference import register_inferer
import typedlist as typedlist_module

import numpy as np

INITIAL_BUFSIZE = 10
SHRINK = 1.5
GROW = 2

def notimplemented(msg):
    raise NotImplementedError("'%s' method of type 'typedlist'" % msg)

_list_cache = {}

#-----------------------------------------------------------------------
# Runtime Constructor
#-----------------------------------------------------------------------

def typedlist(item_type, iterable=None):
    """
    >>> typedlist(int_)
    []
    >>> tlist = typedlist(int_, range(10))
    >>> tlist
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> tlist[5]
    5L

    >>> typedlist(float_, range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    """
    typedlist_ctor = compile_typedlist(item_type)
    return typedlist_ctor(iterable)

#-----------------------------------------------------------------------
# Typedlist implementation
#-----------------------------------------------------------------------

def compile_typedlist(item_type, _list_cache=_list_cache):
    # item_type_t = typesystem.CastType(item_type)
    # dtype_t = typesystem.NumpyDtypeType(item_type)

    if item_type in _list_cache:
        return _list_cache[item_type]

    dtype = item_type.get_dtype()
    methods = orderedcontainer.container_methods(item_type, notimplemented)

    @nb.jit(warn=False)
    class typedlist(object):
        @void(object_)
        def __init__(self, iterable):
            self.size = 0

            # TODO: Use length hint of iterable for initial buffer size
            self.buf = np.empty(INITIAL_BUFSIZE, dtype=dtype)

            # TODO: implement 'is'/'is not'
            if iterable != None:
                self.extend(iterable)

        # TODO: Jit __getitem__/__setitem__ of numba extension types

        __getitem__ = methods['getitem']
        __setitem__ = methods['setitem']
        append = methods['append']
        extend = methods['extend']
        index = methods['index']
        count = methods['count']

        @item_type()
        def pop(self):
            # TODO: Optional argument 'index'
            size = self.size - 1
            item = self.buf[size]
            self.size = size

            if INITIAL_BUFSIZE < size < self.buf.shape[0] / 2:
                self.buf.resize(int(SHRINK * size), refcheck=False)

            return item

        @void(Py_ssize_t, item_type)
        def insert(self, index, value):
            notimplemented("insert")

        @void(item_type)
        def remove(self, value):
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
            notimplemented("sort")

        @Py_ssize_t()
        def __len__(self):
            return self.size

        @nb.c_string_type()
        def __repr__(self):
            buf = ", ".join([str(self.buf[i]) for i in range(self.size)])
            return "[" + buf + "]"

    _list_cache[item_type] = typedlist
    return typedlist

#-----------------------------------------------------------------------
# Infer types for typedlist construction
#-----------------------------------------------------------------------

def typedlist_infer(type_node, iterable_node):
    assert type_node is not None

    type = get_type(type_node)
    if type.is_cast:
        elem_type = type.dst_type

        # Pre-compile typed list implementation
        typedlist_ctor = compile_typedlist(elem_type)

        # Inject the typedlist directly to avoid runtime implementation lookup
        iterable_node = iterable_node or nodes.const(None, object_)
        result = nodes.call_pyfunc(typedlist_ctor, (iterable_node,))
        return nodes.CoercionNode(result, typedlist_ctor.exttype)

    return object_

register_inferer(typedlist_module, 'typedlist', typedlist_infer,
                 pass_in_types=False)


if __name__ == "__main__":
    import doctest
    doctest.testmod()