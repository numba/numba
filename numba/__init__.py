# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
# Import all special functions before registering the Numba module
# type inferer

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from numba.special import *

import os
import sys
import logging
from collections import Iterable

from numba import utils, typesystem

PY3 = sys.version_info[0] == 3

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
from numba.typesystem import *
from numba.minivect.minitypes import FunctionType
from numba.error import *

from numba.containers.typedlist import typedlist
from numba.containers.typedtuple import typedtuple
from numba.typesystem import map_dtype
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


# doctest compatible for jit or autojit numba functions
from numba.tests.test_support import testmod

EXCLUDE_TEST_PACKAGES = ["bytecode"]

def exclude_package_dirs(dirs):
    for exclude_pkg in EXCLUDE_TEST_PACKAGES:
        if exclude_pkg in dirs:
            dirs.remove(exclude_pkg)


def qualified_test_name(root):
    qname = root.replace("/", ".").replace("\\", ".").replace(os.sep, ".") + "."
    offset = qname.rindex('numba.tests.')
    return qname[offset:]

def whitelist_match(whitelist, modname):
    if whitelist:
        return any(item in modname for item in whitelist)
    return True

def map_returncode_to_message(retcode):
    if retcode < 0:
        retcode = -retcode
        return signal_to_name.get(retcode, "Signal %d" % retcode)

    return ""

try:
    import signal
except ImportError:
    signal_to_name = {}
else:
    signal_to_name = dict((signal_code, signal_name)
                           for signal_name, signal_code in vars(signal).items()
                               if signal_name.startswith("SIG"))

def test(whitelist=None, blacklist=None, print_failures_only=False):
    import os
    from os.path import dirname, join
    import subprocess
    import sys

    # FIXME
    # temporarily disable pycc test on win32
    if sys.platform.startswith('win32'):
        blacklist = ['test_pycc_tresult']

    run = failed = 0
    for root, dirs, files in os.walk(join(dirname(__file__), 'tests')):
        qname = qualified_test_name(root)
        exclude_package_dirs(dirs)

        for fn in files:
            if fn.startswith('test_') and fn.endswith('.py'):
                modname, ext = os.path.splitext(fn)
                modname = qname + modname

                if not whitelist_match(whitelist, modname):
                    continue
                if blacklist and whitelist_match(blacklist, modname):
                    continue

                run += 1
                if not print_failures_only:
                    sys.stdout.write("running %-61s" % (modname,))

                process = subprocess.Popen([sys.executable, '-m', modname],
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                out, err = process.communicate()

                if process.returncode == 0:
                    if not print_failures_only:
                        sys.stdout.write("SUCCESS\n")
                else:
                    if print_failures_only:
                        sys.stdout.write("running %-61s" % (modname,))

                    sys.stdout.write("FAILED: %s\n" % map_returncode_to_message(
                                                    process.returncode))
                    if PY3:
                        out = str(out, encoding='UTF-8')
                        err = str(err, encoding='UTF-8')
                    sys.stdout.write(out)
                    sys.stdout.write(err)
                    sys.stdout.write("-" * 80)
                    sys.stdout.write('\n')
                    failed += 1

    sys.stdout.write("ran test files: failed: (%d/%d)\n" % (failed, run))
    return failed

def nose_run(module=None):
    import nose.config
    import __main__

    #os.environ["NOSE_EXCLUDE"] = "(test_all|test_all_noskip|.*compile_with_pycc.*|bytecode)"
    #os.environ["NOSE_VERBOSE"] = "4"

    result = nose.main()
    return len(result.errors), len(result.failures)
