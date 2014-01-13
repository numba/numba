# -*- coding: utf-8 -*-

"""
Graph visualization for abitrary graphs. An attempt to unify all the many
different graphs we want to visualize (numba.control_flow.graphviz,
numba.minivect.graphviz, etc).
"""

from __future__ import print_function, division, absolute_import

import os
import ast
import logging
import textwrap
import subprocess
from itertools import chain

from numba.minivect.pydot import pydot

logger = logging.getLogger(__name__)

#------------------------------------------------------------------------
# Graphviz Mapper
#------------------------------------------------------------------------

class GraphvizGenerator(object):
    """
    Render an arbitrary graph as a graphviz tree.

    Nodes must be hashable.
    """

    counter = 0

    def __init__(self, graph_adaptor, graph_renderer,
                 graph_name, graph_type):
        self.graph_adaptor = graph_adaptor
        self.graph_renderer = graph_renderer

        self.graph = pydot.Dot(graph_name, graph_type=graph_type)

        # Graph Node -> PyDot Node
        self.seen = {}

        # { (source, dest) }
        self.seen_edges = set()

    # ______________________________________________________________________
    # Add to pydot graph

    def add_edge(self, source, dest):
        "Add an edge between two pydot nodes and set the colors"
        if (source, dest) in self.seen_edges:
            return

        self.seen_edges.add((source, dest))
        edge = pydot.Edge(self.seen[source], self.seen[dest])

        edge_label = self.graph_renderer.render_edge(source, dest)
        if edge_label is not None:
            edge.set_label(edge_label)

        self.graph.add_edge(edge)

    def create_node(self, node):
        "Create a graphviz node from the miniast node"
        label = self.graph_renderer.render(node)
        self.counter += 1
        pydot_node = pydot.Node(str(self.counter), label=label, shape='box')
        self.graph.add_node(pydot_node)
        return pydot_node

    # ______________________________________________________________________
    # Traverse Graph

    def dfs(self, node):
        "Visit children and add edges to their Graphviz nodes."
        if node in self.seen:
            return

        pydot_node = self.create_node(node)
        self.seen[node] = pydot_node

        for child in self.graph_adaptor.children(node):
            self.dfs(child)
            self.add_edge(node, child)

#------------------------------------------------------------------------
# Graph Adaptors
#------------------------------------------------------------------------

class GraphAdaptor(object):
    """
    Allow traversal of a foreign AST.
    """

    def children(self, node):
        "Return the children for this graph node"

#------------------------------------------------------------------------
# Graph Rendering
#------------------------------------------------------------------------

class GraphRenderer(object):
    """
    Allow traversal of a foreign AST.
    """

    def render(self, node):
        "Return the label for this graph node"

    def render_edge(self, source, dest):
        "Return the label for this edge or None"


#------------------------------------------------------------------------
# Create image from dot
#------------------------------------------------------------------------

def write_image(dot_output):

    prefix, ext = os.path.splitext(dot_output)
    png_output = prefix + '.png'

    fp = open(png_output, 'wb')
    try:
        p = subprocess.Popen(['dot', '-Tpng', dot_output],
                             stdout=fp.fileno(),
                             stderr=subprocess.PIPE)
        p.wait()
    except EnvironmentError as e:
        logger.warn("Unable to write png: %s (did you install the "
                    "'dot' program?). Wrote %s" % (e, dot_output))
    else:
        logger.warn("Wrote %s" % png_output)
    finally:
        fp.close()

#------------------------------------------------------------------------
# Entry points
#------------------------------------------------------------------------

def render(G, output_file, adaptor, renderer,
           graph_name="G", graph_type="digraph"):
    """
    G: The graph: [node]
    output_file: output dot file name
    adaptor: GraphAdaptor
    renderer: GraphRenderer
    """
    gen = GraphvizGenerator(adaptor, renderer, graph_name, graph_type)
    for root in G:
        gen.dfs(root)

    dotgraph = gen.graph
    # output_file, ext = os.path.splitext(output_file)
    # dotgraph.write(output_file + '.png', format='png')
    dotgraph.write(output_file)
    write_image(output_file)
