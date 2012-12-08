### Copyright (2012) Continuum Analytics, Inc
### All Rights Reserved

import sys, os, logging
import subprocess, tempfile
import functools

import numba.pycc as pyc

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_ending(args):  
    if args.llvm:
        return ".bc"
    if args.olibs:
        return ".o"
    else:
        return pyc.find_shared_ending()

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
    parser.add_argument('--python', action='store_true',
                        help='Emit additionally generated Python wrapper and '
                        'extension module code in output')

    if os.path.basename(args[0]) in ['pycc.py', 'pycc']:
        args = args[1:]

    args = parser.parse_args(args)
    if args.output:
        args.output = args.output[0]
        output_base = os.path.split(args.output)[1]
        module_name = os.path.splitext(output_base)[0]
    else:
        input_base = os.path.splitext(args.inputs[0])[0]
        module_name = os.path.split(input_base)[1]
        args.output = input_base + get_ending(args)
    logger.debug('args.output --> %s', args.output)

    # run the compiler
    logger.debug('inputs --> %s', args.inputs)
    compiler = pyc.Compiler(args.inputs, module_name=module_name)
    if args.llvm:
        logger.debug('emit llvm')
        compiler.write_llvm_bitcode(args.output, wrap=args.python)
    elif args.olibs:
        logger.debug('emit object file')
        compiler.write_native_object(args.output, wrap=args.python)
    else:
        logger.debug('emit shared library')
        logger.debug('write to temporary object file %s', tempfile.gettempdir())
        temp_obj = (tempfile.gettempdir() + os.sep +
                    os.path.basename(args.output) + '.o')
        compiler.write_native_object(temp_obj, wrap=args.python)
        cmdargs = (pyc.find_linker(),) + pyc.find_args() + ('-o', args.output, temp_obj)
        subprocess.check_call(cmdargs)
        os.remove(temp_obj)   # remove temporary object


    if args.header:
        compiler.emit_header(args.output)

