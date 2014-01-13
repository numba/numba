# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# Import all special functions before registering the Numba module
# type inferer
from numba.special import *

import os
import sys
import logging

PY3 = sys.version_info[0] == 3

from numba import utils, typesystem

def get_include():
    numba_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(numba_root, "include")

# NOTE: Be sure to keep the logging level commented out before commiting.  See:
#   https://github.com/numba/numba/issues/31
# A good work around is to make your tests handle a debug flag, per
# numba.tests.test_support.main().

class _RedirectingHandler(logging.Handler):
    '''
    A log hanlder that applies its formatter and redirect the emission
    to a parent handler.
    '''
    def set_handler(self, handler):
        self.handler = handler

    def emit(self, record):
        # apply our own formatting
        record.msg = self.format(record)
        record.args = [] # clear the args
        # use parent handler to emit record
        self.handler.emit(record)

def _config_logger():
    root = logging.getLogger(__name__)
    format = "\n\033[1m%(levelname)s -- "\
             "%(module)s:%(lineno)d:%(funcName)s\033[0m\n%(message)s"
    try:
        parent_hldr = root.parent.handlers[0]
    except IndexError: # parent handler is not initialized?
        # build our own handler --- uses sys.stderr by default.
        parent_hldr = logging.StreamHandler()
    hldr = _RedirectingHandler()
    hldr.set_handler(parent_hldr)
    fmt = logging.Formatter(format)
    hldr.setFormatter(fmt)
    root.addHandler(hldr)
    root.propagate = False # do not propagate to the root logger

_config_logger()


from . import special
from numba.typesystem import template
from numba.typesystem import *
from numba.typesystem import struct_ as struct # don't export this in __all__
from numba.typesystem import function
from numba.error import *

from numba.containers.typedlist import typedlist
from numba.containers.typedtuple import typedtuple
from numba.typesystem.numpy_support import map_dtype
from numba.type_inference.module_type_inference import (is_registered,
                                                        register,
                                                        register_inferer,
                                                        get_inferer,
                                                        register_unbound,
                                                        register_callable)
from numba.typesystem.typeset import *

from numba.codegen import translate
from numba.decorators import *
from numba import decorators
from numba.intrinsic.numba_intrinsic import (declare_intrinsic,
                                             declare_instruction)

__all__ = typesystem.__all__ + decorators.__all__ + special.__all__
__all__.extend(["numeric", "floating", "complextypes"])

from numba import testing
from numba.testing import test, nose_run
from numba.testing.user_support import testmod
