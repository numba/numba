from __future__ import print_function, absolute_import

import functools
from .manager import DataModelManager


def register(dmm, typecls):
    def wraps(fn):
        dmm.register(typecls, fn)
        return fn

    return wraps


defaultDataModelManager = DataModelManager()

register_default = functools.partial(register, defaultDataModelManager)
