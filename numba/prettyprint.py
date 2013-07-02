# -*- coding: utf-8 -*-

"""
Pretty printing of numba IRs.
"""

from __future__ import print_function, division, absolute_import

import os
import sys

from numba.lexing import lex_source
from numba.viz import cfgviz, astviz
from numba.annotate import annotators
from numba.annotate import render_text, render_html
from numba.annotate.annotate import Source, Program, build_linemap

# ______________________________________________________________________

def dumppass(option):
    def decorator(f):
        def wrapper(ast, env):
            if env.cmdopts.get(option):
                f(ast, env, env.cmdopts.get("fancy"))
            return ast
        return wrapper
    return decorator

# ______________________________________________________________________

@dumppass("dump-ast")
def dump_ast(ast, env, fancy):
    if fancy:
        astviz.render_ast(ast, os.path.expanduser("~/ast.dot"))
    else:
        import ast as ast_module
        print(ast_module.dump(ast))

@dumppass("dump-cfg")
def dump_cfg(ast, env, fancy):
    cfg = env.crnt.flow
    if fancy:
        cfgviz.render_cfg(cfg, os.path.expanduser("~/cfg.dot"))
    else:
        for block in cfg.blocks:
            print(block)
            print("    ", block.parents)
            print("    ", block.children)

@dumppass("annotate")
def dump_annotations(ast, env, fancy):
    llvm_intermediate, = [i for i in env.crnt.intermediates if i.name == "llvm"]
    annotators.annotate_pyapi(llvm_intermediate, env.crnt.annotations)

    p = Program(Source(build_linemap(env.crnt.func), env.crnt.annotations),
                env.crnt.intermediates)
    if fancy:
        render = render_html
        fn, ext = os.path.splitext(env.cmdopts["filename"])
        out = open(fn + '.html', 'w')
        print("Writing", fn + '.html')
    else:
        render = render_text
        out = sys.stdout

    # Look back through stack until we find the line of code that called our
    # jitted function.
    func_call = ''
    func_call_filename = env.cmdopts['filename']
    func_call_lineno = ''
    import traceback
    stack = traceback.extract_stack()
    for i in range(len(stack)-1, -1, -1):
        if stack[i][0] == env.cmdopts['filename'] and stack[i][3].find(env.crnt.func_name) > -1:
            func_call = stack[i][3]
            func_call_lineno = str(stack[i][1])
            break

    annotation = {'func_call':func_call,
                  'func_call_filename':func_call_filename,
                  'func_call_lineno':func_call_lineno,
                  'python_source':p.python_source,
                  'intermediates':p.intermediates}
    if fancy:
        env.annotation_blocks.append(annotation)
    else:
        env.annotation_blocks = [annotation]

    render(env.annotation_blocks, emit=out.write, intermediate_names=["llvm"])

@dumppass("dump-llvm")
def dump_llvm(ast, env, fancy):
    print(lex_source(str(env.crnt.lfunc), "llvm", "console"))

@dumppass("dump-optimized")
def dump_optimized(ast, env, fancy):
    print(lex_source(str(env.crnt.lfunc), "llvm", "console"))