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
