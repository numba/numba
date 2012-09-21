#!/usr/bin/env python
### Copyright (2012) Continuum Analytics, Inc
### All Rights Reserved

"""
  pyc --- python compiler
  Produce a shared library from Python code

  Usage:  pyc <input-py-file(s)> -o output-file

  Compile input files to a single shared library (final step uses the platform
  linker which must be installed).

  Options:
             -h  Help
             -o  Name of output file (default is name of first input -- with new ending)
             -c  Create object code from each input file instead of shared library
             --llvm Emit llvm module instead of object code
             --linker path-to-linker (if not on $PATH and llvm not provided)
             --linker-args string of args (be sure to use quotes)
             --headers output header files
"""
import sys, os, logging
import subprocess, tempfile
import pyc_internal as pyc

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

_configs = {'win' : ("link.exe", "/dll", '.dll'),
	'dar': ("libtool", "-dynamic", '.dylib'),
 	'default': ("ld", "-shared", ".so")
}

def get_configs(arg):
    return _configs.get(sys.platform[:3], _configs['default'])[arg]

import functools

find_linker = functools.partial(get_configs, 0)
find_args = functools.partial(get_configs, 1)
find_shared_ending = functools.partial(get_configs, 2)

def get_ending(args):  
    if args.llvm:
        return ".bc"
    if args.olibs:
        return ".o"
    else:
        return find_shared_ending()

def parse_arguments(args):
    inputs = args.inputs
    args.output = args.output[0] if args.output else os.path.splitext(inputs[0])[0] + get_ending(args)
    logger.debug('args.output --> %s', args.output)
    return args

def main(args=[]):
    if not args:
        args = sys.argv
    import argparse
    parser = argparse.ArgumentParser(description="Compile Python modules to a single shared library")
    parser.add_argument("inputs", nargs='+', help="Input file(s)")
    parser.add_argument("-o", nargs=1, dest="output", help="Output file")
    parser.add_argument("-c", action="store_true", dest="olibs",
                        help="Create object file from each input instead of shared-library")
    parser.add_argument("--llvm", action="store_true",
                        help="Emit llvm instead of native code")
    parser.add_argument("--linker", nargs=1, help="Path to linker (default is platform dependent)")
    parser.add_argument("--linker-args", help="Arguments to pass to linker")
    parser.add_argument('--header', action="store_true",
                        help="Emit C header file with function signatures")

    if os.path.basename(args[0]) in ['pyc.py', 'pyc']:
        args = args[1:]

    argattr = parse_arguments(parser.parse_args(args))

    # run the compiler
    logger.debug('inputs --> %s', argattr.inputs)
    compiler = pyc.Compiler(argattr.inputs)
    if argattr.llvm:
        logger.debug('emit llvm')
        compiler.write_llvm_bitcode(argattr.output)
    elif argattr.olibs:
        logger.debug('emit object file')
        compiler.write_native_object(argattr.output)
    else:
        logger.debug('emit shared library')
        logger.debug('write to temporary object file %s', tempfile.gettempdir())
        temp_obj = tempfile.gettempdir() + os.sep + os.path.basename(argattr.output) + '.o'
        temp_obj = os.path.basename(argattr.output)+'.o'
        compiler.write_native_object(temp_obj)          # write temporary object
        cmdargs = find_linker(), find_args(), '-o', argattr.output, temp_obj
        subprocess.check_call(cmdargs)
        os.remove(temp_obj)   # remove temporary object

    if argattr.header:
        pyc.emit_header(argattr.output)

if __name__ == "__main__":
    main()

