from __future__ import print_function, absolute_import

import numbers


class KernelRuntimeError(RuntimeError):
    def __init__(self, msg, tid=None, ctaid=None):
        self.tid = tid
        self.ctaid = ctaid
        self.msg = msg
        t = ("An exception was raised in thread=%s block=%s\n"
             "\t%s")
        msg = t % (self.tid, self.ctaid, self.msg)
        super(KernelRuntimeError, self).__init__(msg)


def normalize_kernel_dimensions(griddim, blockdim):
    """
    Normalize and validate the user-supplied kernel dimensions.
    """

    def check_dim(dim, name):
        if not isinstance(dim, (tuple, list)):
            dim = [dim]
        else:
            dim = list(dim)
        if len(dim) > 3:
            raise ValueError('%s must be a sequence of 1, 2 or 3 integers, got %r'
                             % (name, dim))
        for v in dim:
            if not isinstance(v, numbers.Integral):
                raise TypeError('%s must be a sequence of integers, got %r'
                                % (name, dim))
        while len(dim) < 3:
            dim.append(1)
        return dim

    griddim = check_dim(griddim, 'griddim')
    blockdim = check_dim(blockdim, 'blockdim')

    return griddim, blockdim
