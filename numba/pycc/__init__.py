# -*- coding: utf-8 -*-


import os
import logging
import subprocess
import tempfile
import sys

# Public API
from .cc import CC
from .decorators import export, exportmany


def get_ending(args):
    if args.llvm:
        return ".bc"
    elif args.olibs:
        return ".o"
    elif args.python:
        return find_pyext_ending()
    else:
        return find_shared_ending()


def main(args=None):
    import argparse

    from .compiler import ModuleCompiler
    from .platform import Toolchain, find_shared_ending, find_pyext_ending
    from numba.pycc import decorators

    parser = argparse.ArgumentParser(
        description="DEPRECATED - Compile Python modules to a single shared library")
    parser.add_argument("inputs", nargs='+', help="Input file(s)")
    parser.add_argument("-o", nargs=1, dest="output",
                        help="Output file  (default is name of first input -- with new ending)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", action="store_true", dest="olibs",
                       help="Create object file from each input instead of shared-library")
    group.add_argument("--llvm", action="store_true",
                       help="Emit llvm instead of native code")

    parser.add_argument('--header', action="store_true",
                        help="Emit C header file with function signatures")
    parser.add_argument('--python', action='store_true',
                        help='Emit additionally generated Python wrapper and '
                        'extension module code in output')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Print extra debug information')

    args = parser.parse_args(args)

    logger = logging.getLogger(__name__)
    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.warn("The 'pycc' script is DEPRECATED; "
                "please use the numba.pycc.CC API instead")

    if args.output:
        args.output = args.output[0]
        output_base = os.path.split(args.output)[1]
        module_name = os.path.splitext(output_base)[0]
    else:
        input_base = os.path.splitext(args.inputs[0])[0]
        module_name = os.path.split(input_base)[1]
        args.output = input_base + get_ending(args)
    logger.debug('args.output --> %s', args.output)

    if args.header:
        print('ERROR: pycc --header has been disabled in this release due to a known issue')
        sys.exit(1)

    logger.debug('inputs --> %s', args.inputs)
    decorators.process_input_files(args.inputs)

    compiler = ModuleCompiler(decorators.export_registry, module_name=module_name)
    if args.llvm:
        logger.debug('emit llvm')
        compiler.write_llvm_bitcode(args.output, wrap=args.python)
    elif args.olibs:
        logger.debug('emit object file')
        compiler.write_native_object(args.output, wrap=args.python)
    else:
        logger.debug('emit shared library')
        logger.debug('write to temporary object file %s', tempfile.gettempdir())

        toolchain = Toolchain()
        toolchain.debug = args.debug
        temp_obj = (tempfile.gettempdir() + os.sep +
                    os.path.basename(args.output) + '.o')
        compiler.write_native_object(temp_obj, wrap=args.python)
        libraries = toolchain.get_python_libraries()
        toolchain.link_shared(args.output, [temp_obj],
                              toolchain.get_python_libraries(),
                              toolchain.get_python_library_dirs(),
                              export_symbols=compiler.dll_exports)
        os.remove(temp_obj)
