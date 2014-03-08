from __future__ import print_function, absolute_import, division


class CudaDriverError(Exception):
    pass


class CudaSupportError(ImportError):
    pass


class NvvmError(Exception):
    pass


class NvvmSupportError(ImportError):
    pass
