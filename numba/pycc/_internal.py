import logging
import os
import sys
import functools
from importlib import import_module
from numba import decorators
logger = logging.getLogger(__name__)

__all__ = ['which', 'find_linker', 'find_args', 'find_shared_ending', 'Compiler',
           'emit_header']

def which(program):
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, fname)
            if is_exe(exe_file):
                return exe_file
    return None

_configs = {'win' : ("link.exe", ("/dll",), '.dll'),
    'dar': ("libtool", ("-dynamic",), '.dylib'),
    'default': ("ld", ("-shared",), ".so")
}

def get_configs(arg):
    return _configs.get(sys.platform[:3], _configs['default'])[arg]

find_linker = functools.partial(get_configs, 0)
find_args = functools.partial(get_configs, 1)
find_shared_ending = functools.partial(get_configs, 2)


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
        execfile(ifile)
        lmod = decorators.default_module
        return lmod

def emit_header(output):
    from numba.minivect import minitypes

    fname, ext = os.path.splitext(output)
    with open(fname + '.h', 'wb') as fout:
        fout.write(minitypes.get_utility())
        fout.write("\n/* Prototypes */\n")
        for t, name in decorators.translated:
            name = name or t.func.func_name
            restype = t.mini_rettype.declare()
            args = ", ".join(arg_type.declare() for arg_type in t.mini_argtypes)
            fout.write("extern %s %s(%s);\n" % (restype, name, args))
