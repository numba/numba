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
             -o  Name of output file (default is name of first input with extension changed)
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

def find_linker():
    if sys.platform.startswith('win'):
        return which("link.exe")
    else:
        return which("ld")

def get_ending(args):
    import distutils.sysconfig
    if args.llvm:
        return ".bc"
    if args.olibs:
        return ".o"
    else:
        return distutils.sysconfig.get_config_var('SO')

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
        compiler.write_native_object(temp_obj)          # write temporary object
        cmdargs = find_linker(), '-shared', '-o', argattr.output, temp_obj
        subprocess.check_call(cmdargs)
        os.remove(temp_obj)   # remove temporary object

if __name__ == "__main__":
    main()

