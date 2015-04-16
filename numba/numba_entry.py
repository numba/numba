from __future__ import print_function, division, absolute_import

import sys
import argparse
import os
import subprocess


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotate', help='Annotate source',
                        action='store_true')
    parser.add_argument('--dump-llvm', action="store_true",
                        help='Print generated llvm assembly')
    parser.add_argument('--dump-optimized', action='store_true',
                        help='Dump the optimized llvm assembly')
    parser.add_argument('--dump-assembly', action='store_true',
                        help='Dump the LLVM generated assembly')
    parser.add_argument('--dump-cfg', action="store_true",
                        help='[Deprecated] Dump the control flow graph')
    parser.add_argument('--dump-ast', action="store_true",
                        help='[Deprecated] Dump the AST')
    parser.add_argument('--annotate-html', nargs=1,
                        help='Output source annotation as html')
    parser.add_argument('filename', help='Python source filename')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.dump_cfg:
        print("CFG dump is removed.")
        sys.exit(1)
    if args.dump_ast:
        print("AST dump is removed.  Numba no longer depends on AST.")
        sys.exit(1)

    os.environ['NUMBA_DUMP_ANNOTATION'] = str(int(args.annotate))
    if args.annotate_html is not None:
        try:
            from jinja2 import Template
        except ImportError:
            raise ImportError("Please install the 'jinja2' package")
        os.environ['NUMBA_DUMP_HTML'] = str(args.annotate_html[0])
    os.environ['NUMBA_DUMP_LLVM'] = str(int(args.dump_llvm))
    os.environ['NUMBA_DUMP_OPTIMIZED'] = str(int(args.dump_optimized))
    os.environ['NUMBA_DUMP_ASSEMBLY'] = str(int(args.dump_assembly))

    cmd = [sys.executable, args.filename]
    subprocess.call(cmd)


