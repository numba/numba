# -*- coding: utf-8 -*-

"""
Visualize an AST.
"""

from __future__ import print_function, division, absolute_import

import os
import ast
import textwrap
from itertools import chain, imap, ifilter

from numba.viz.graphviz import render

# ______________________________________________________________________
# AST Constants

ast_constant_classes = (
    ast.expr_context,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
)

# ______________________________________________________________________
# Utilities

is_ast = lambda node: (isinstance(node, (ast.AST, list)) and not
                       isinstance(node, ast_constant_classes))

class SomeConstant(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return repr(self.value)

def make_list(node):
    if isinstance(node, list):
        return node
    elif isinstance(node, ast.AST):
        return [node]
    else:
        return [SomeConstant(node)]

def nodes(node):
    return [getattr(node, attr, None) for attr in node._fields]

def fields(node):
    return zip(node._fields, nodes(node))

# ______________________________________________________________________
# Adaptor

class ASTGraphAdaptor(object):

    def children(self, node):
        return list(chain(*imap(make_list, ifilter(is_ast, nodes(node)))))

# ______________________________________________________________________
# Renderer

def strval(val):
    if isinstance(val, ast_constant_classes):
        return type(val).__name__ # Load, Store, Param
    else:
        return repr(val)

class ASTGraphRenderer(object):

    def render(self, node):
        all_fields = fields(node)

        for attr in getattr(node, '_attributes', []):
            if attr not in node._fields and hasattr(node, attr):
                all_fields.append((attr, getattr(node, attr)))

        all_fields = [(attr, v) for attr, v in all_fields if not is_ast(v)]
        args = ",\n".join('%s=%s' % (a, strval(v)) for a, v in all_fields)
        if args:
            args = '\n' + args
        return "%s(%s)" % (type(node).__name__, args)

    def render_edge(self, source, dest):
        # See which attribute of the source node matches the destination node
        for attr_name, attr in fields(source):
            if attr is dest or (isinstance(attr, list) and dest in attr):
                # node.attr == dst_node or dest_node in node.attr
                return attr_name

# ______________________________________________________________________
# Entry Point

def render_ast(ast, output_file):
    render([ast], output_file, ASTGraphAdaptor(), ASTGraphRenderer())

# ______________________________________________________________________
# Test

if __name__ == '__main__':
    source = textwrap.dedent("""
        def func(a, b):
            for i in range(10):
                if i < 5:
                    print "hello"
    """)
    mod = ast.parse(source)
    print(mod)
    render_ast(mod, os.path.expanduser("~/ast.dot"))