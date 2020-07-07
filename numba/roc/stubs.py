from numba.core.rewrites.macros import Macro
from numba.core import types, typing, ir

_stub_error = NotImplementedError("This is a stub.")


def get_global_id(*args, **kargs):
    """
    OpenCL get_global_id()
    """
    raise _stub_error


def get_local_id(*args, **kargs):
    """
    OpenCL get_local_id()
    """
    raise _stub_error


def get_global_size(*args, **kargs):
    """
    OpenCL get_global_size()
    """
    raise _stub_error


def get_local_size(*args, **kargs):
    """
    OpenCL get_local_size()
    """
    raise _stub_error


def get_group_id(*args, **kargs):
    """
    OpenCL get_group_id()
    """
    raise _stub_error


def get_num_groups(*args, **kargs):
    """
    OpenCL get_num_groups()
    """
    raise _stub_error


def get_work_dim(*args, **kargs):
    """
    OpenCL get_work_dim()
    """
    raise _stub_error


def barrier(*args, **kargs):
    """
    OpenCL barrier()

    Example:

        # workgroup barrier + local memory fence
        hsa.barrier(hsa.CLK_LOCAL_MEM_FENCE)
        # workgroup barrier + global memory fence
        hsa.barrier(hsa.CLK_GLOBAL_MEM_FENCE)
        # workgroup barrier + global memory fence
        hsa.barrier()

    """
    raise _stub_error


def mem_fence(*args, **kargs):
    """
    OpenCL mem_fence()

    Example:

        # local memory fence
        hsa.mem_fence(hsa.CLK_LOCAL_MEM_FENCE)
        # global memory fence
        hsa.mem_fence(hsa.CLK_GLOBAL_MEM_FENCE)
    """
    raise _stub_error


def wavebarrier():
    """
    HSAIL wavebarrier
    """
    raise _stub_error


def activelanepermute_wavewidth(src, laneid, identity, useidentity):
    """
    HSAIL activelanepermute_wavewidth_*
    """
    raise _stub_error


def ds_permute(src_lane, dest_lane):
    """
    AMDGCN Data Share intrinsic forwards permute (push semantics)
    """
    raise _stub_error


def ds_bpermute(src_lane, dest_lane):
    """
    AMDGCN Data Share intrinsic backwards permute (pull semantics)
    """
    raise _stub_error


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

    array = Macro('shared.array', shared_array, callable=True,
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
