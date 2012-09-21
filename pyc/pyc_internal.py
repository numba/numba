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

    def write_llvm_bitcode(self, output):
        for ifile in self.inputs:
            self.compile_to_default_module(ifile)

        lmod = decorators.default_module
        with open(output, 'wb') as fout:
            lmod.to_bitcode(fout)

    def write_native_object(self, output):
        for ifile in self.inputs:
            self.compile_to_default_module(ifile)

        lmod = decorators.default_module
        with open(output, 'wb') as fout:
            fout.write(lmod.to_native_object())

    def compile_to_default_module(self, ifile):
        modname = _filepath_to_module(ifile)
        logger.debug('module name %s', modname)
        pymod = import_module(modname)
        logger.debug(pymod)
        lmod = decorators.default_module
        return lmod

def emit_header(output):
    fname, ext = os.path.splitext(output)
    with open(fname + '.h', 'wb') as fout:
        for t in decorators.translates:
            name = t.func.func_name
            ret_type = str(t.ret_type)
            args = ", ".join(str(arg_type) for arg_type in t.arg_types)
            fout.write("extern %s %s(%s);\n" % (ret_type, name, args))