from __future__ import print_function, absolute_import
from numba import types, ir, typing, macro


def get_global_id(*args, **kargs):
    """
    OpenCL get_global_id()
    """
    raise NotImplementedError("This is a stub.")


def get_local_id(*args, **kargs):
    """
    OpenCL get_local_id()
    """
    raise NotImplementedError("This is a stub.")


def get_global_size(*args, **kargs):
    """
    OpenCL get_global_size()
    """
    raise NotImplementedError("This is a stub.")


def get_local_size(*args, **kargs):
    """
    OpenCL get_local_size()
    """
    raise NotImplementedError("This is a stub.")


def get_group_id(*args, **kargs):
    """
    OpenCL get_group_id()
    """
    raise NotImplementedError("This is a stub.")


def get_num_groups(*args, **kargs):
    """
    OpenCL get_num_groups()
    """
    raise NotImplementedError("This is a stub.")


def get_work_dim(*args, **kargs):
    """
    OpenCL get_work_dim()
    """
    raise NotImplementedError("This is a stub.")


def barrier(*args, **kargs):
    """
    OpenCL barrier()
    """
    raise NotImplementedError("This is a stub.")


def mem_fence(*args, **kargs):
    """
    OpenCL mem_fence()
    """
    raise NotImplementedError("This is a stub.")


class Stub(object):
    """A stub object to represent special objects which is meaningless
    outside the context of HSA-python.
    """
    _description_ = '<ptx special value>'
    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_


def shared_array(shape, dtype):
    shape = _legalize_shape(shape)
    ndim = len(shape)
    fname = "hsail.smem.alloc"
    restype = types.Array(dtype, ndim, 'C')
    sig = typing.signature(restype, types.UniTuple(types.intp, ndim), types.Any)
    return ir.Intrinsic(fname, sig, args=(shape, dtype))


class shared(Stub):
    """shared namespace
    """
    _description_ = '<shared>'

    array = macro.Macro('shared.array', shared_array, callable=True,
                        argnames=['shape', 'dtype'])


def _legalize_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise TypeError("invalid type for shape; got {0}".format(type(shape)))

#-------------------------------------------------------------------------------
# atomic

class atomic(Stub):
    """atomic namespace
    """
    _description_ = '<atomic>'

    class add(Stub):
        """add(ary, idx, val)

        Perform atomic ary[idx] += val
        """
