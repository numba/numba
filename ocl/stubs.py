from __future__ import print_function, absolute_import
from numba import types, ir, typing, macro


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
    """
    raise _stub_error


def mem_fence(*args, **kargs):
    """
    OpenCL mem_fence()
    """
    raise _stub_error


def sub_group_barrier():
    """
    OpenCL 2.0 sub_group_barrier
    """
    raise _stub_error


class Stub(object):
    """A stub object to represent special objects which is meaningless
    outside the context of DPPy compilation context.
    """
    _description_ = '<dppy special value>'
    __slots__ = ()  # don't allocate __dict__

    def __new__(cls):
        raise NotImplementedError("%s is not instantiable" % cls)

    def __repr__(self):
        return self._description_

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

    class sub(Stub):
        """sub(ary, idx, val)

        Perform atomic ary[idx] -= val
        """
