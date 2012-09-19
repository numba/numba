import logging
import os
from importlib import import_module
from numbapro import decorators
logger = logging.getLogger(__name__)

def _filepath_to_module(filepath):
    name, _ext = filepath.split('.', 1)
    return name.replace(os.sep, '.')

class Compiler(object):
    def __init__(self, inputs):
        self.inputs = inputs

    def emit_llvm(self, output):
        for ifile in self.inputs:
            ofile = '%s.ll' % ifile
            logger.debug('compiling %s --> %s', ifile, ofile)
            modname = _filepath_to_module(ifile)
            logger.debug('module name %s', modname)
            pymod = import_module(modname)
            logger.debug(pymod)
            lmod = decorators.default_module
            print lmod
