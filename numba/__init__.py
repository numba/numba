import sys
import logging

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

from . import _numba_types
from ._numba_types import *
from . import decorators
from .decorators import *

def test():
    import os
    from os.path import dirname, join
    from subprocess import call

    run = failed = 0
    for fn in os.listdir(join(dirname(__file__), 'tests')):
        if fn.endswith('.py'):
            modname = fn[:-3]
            run += 1
            res = call([sys.executable, '-m', 'numba.tests.' + modname])
            if res != 0:
                failed += 1
    print "ran test files: failed: (%d/%d)" % (failed, run)
    return failed

def nose_run():
    "Oh nose, why dost thou never read my configuration file"
    import nose.config
    config = nose.config.Config()
    config.configure(['--logging-level=DEBUG',
                      '--verbosity=3',      # why is this ignored?
                      # '--with-doctest=1', # not recognized?
                      #'--doctest-tests'
                      ])
    config.verbosity = 3
    nose.run(config=config)

__all__ = _numba_types.__all__ + decorators.__all__
