# -*- coding: utf-8 -*-

"""
Visualize a CFG.
"""

from __future__ import print_function, division, absolute_import

import os
import ast
import textwrap

from numba import void
from numba import utils
from numba.viz import graphviz

# ______________________________________________________________________
# Utilities

def cf_from_source(source, func_globals):
    "Render the SSA graph given python source code"
    from numba import pipeline
    from numba import environment

    mod = ast.parse(source)
    func_ast = mod.body[0]

    env = environment.NumbaEnvironment.get_environment()
    func_env, _ = pipeline.run_pipeline2(env, None, func_ast, void(),
                                         pipeline_name='cf',
                                         function_globals=dict(func_globals))
    return func_env.symtab, func_env.flow #func_env.cfg

# ______________________________________________________________________
# Adaptor

class CFGGraphAdaptor(graphviz.GraphAdaptor):
    def children(self, node):
        return node.children

# ______________________________________________________________________
# Renderer

def fmtnode(node):
    if isinstance(node, ast.Assign):
        return "%s = %s" % (node.targets, node.value)
    else:
        return str(node)

fmtnode = utils.pformat_ast # ast.dump # str

class CFGGraphRenderer(graphviz.GraphRenderer):

    def render(self, node):
        return "%s\n%s" % (node, "; ".join(map(fmtnode, node.body)))

# ______________________________________________________________________
# Entry Points

def render_cfg(cfflow, output_file):
    "Render the SSA graph given the flow.CFGFlow and the symbol table"
    graphviz.render(cfflow.blocks, output_file,
                    CFGGraphAdaptor(), CFGGraphRenderer())

def render_cfg_from_source(source, output_file, func_globals=()):
    "Render the SSA graph given python source code"
    symtab, cfflow = cf_from_source(source, func_globals)
    render_cfg(cfflow, output_file)

# ______________________________________________________________________
# Test

if __name__ == '__main__':
    source = textwrap.dedent("""
        def func():
            # x_0
            x = 0 # x_1
            # x_2
            for i in range(10):
                if i < 5:
                    x = i # x_3
                # x_4
                x = x + i # x_5

            y = x
            x = i # x_6
    """)
    render_cfg_from_source(source, os.path.expanduser("~/cfg.dot"))