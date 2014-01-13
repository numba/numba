# -*- coding: utf-8 -*-

"""
Visualize an SSA graph (the def/use chains).
"""

from __future__ import print_function, division, absolute_import

import os
import textwrap

from numba.viz.graphviz import render
from numba.viz.cfgviz import cf_from_source
from numba.control_flow.cfstats import NameAssignment

# ______________________________________________________________________
# Adaptor

class SSAGraphAdaptor(object):
    def children(self, node):
        return [nameref.variable for nameref in node.cf_references]

# ______________________________________________________________________
# Renderer

class SSAGraphRenderer(object):

    def render(self, node):
        if node.renamed_name:
            return node.unmangled_name
        return node.name

    def render_edge(self, source, dest):
        return "use"

# ______________________________________________________________________
# Entry Points

def render_ssa(cfflow, symtab, output_file):
    "Render the SSA graph given the flow.CFGFlow and the symbol table"
    cfstats = [stat for b in cfflow.blocks for stat in b.stats]
    defs = [stat.lhs.variable for stat in cfstats
                                  if isinstance(stat, NameAssignment)]
    nodes = symtab.values() + defs
    render(nodes, output_file, SSAGraphAdaptor(), SSAGraphRenderer())

def render_ssa_from_source(source, output_file, func_globals=()):
    "Render the SSA graph given python source code"
    symtab, cfflow = cf_from_source(source, func_globals)
    render_ssa(cfflow, symtab, output_file)

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
                # x = x + i # x_5

            y = x
            x = i # x_6
    """)

    render_ssa_from_source(source, os.path.expanduser("~/ssa.dot"))