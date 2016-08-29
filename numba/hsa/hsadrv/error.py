from __future__ import print_function, absolute_import, division


class HsaDriverError(Exception):
    pass


class HsaSupportError(ImportError):
    pass


class HsaApiError(HsaDriverError):
    def __init__(self, code, msg):
        self.code = code
        super(HsaApiError, self).__init__(msg)


class HsaWarning(UserWarning):
    pass


class HsaKernelLaunchError(HsaDriverError):
    pass


class HsaContextMismatchError(HsaDriverError):
    def __init__(self, expect, got):
        msg = ("device array is associated with a different "
               "context: expect {0} but got {1}")
        super(HsaContextMismatchError, self).__init__(msg.format())

