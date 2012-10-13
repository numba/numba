"""
Visitor to generate a Graphviz .dot file with an AST representation.
"""

from pydot import pydot

import minivisitor

class GraphvizGenerator(minivisitor.PrintTree):
    """
    Render a minivect AST as a graphviz tree.
    """

    def __init__(self, context, name, node_color=None, edge_color=None,
                 node_fontcolor=None, edge_fontcolor=None):
        super(GraphvizGenerator, self).__init__(context)
        self.name = name
        self.counter = 0
        self.node_color = node_color
        self.edge_color = edge_color
        self.node_fontcolor = node_fontcolor
        self.edge_fontcolor = edge_fontcolor

    def create_node(self, node):
        "Create a graphviz node from the miniast node"
        label = '"%s"' % self.format_node(node, want_type_info=False)
        self.counter += 1
        pydot_node = pydot.Node(str(self.counter), label=label)
        self.add_node(pydot_node)
        return pydot_node

    def add_node(self, pydot_node):
        "Add a pydot node to the graph and set its colors"
        if self.node_color is not None:
            pydot_node.set_color(self.node_color)
        if self.node_fontcolor is not None:
            pydot_node.set_fontcolor(self.node_fontcolor)

        self.graph.add_node(pydot_node)

    def add_edge(self, source, dest, edge_label=None):
        "Add an edge between two pydot nodes and set the colors"
        edge = pydot.Edge(source, dest)

        if edge_label is not None:
            edge.set_label(edge_label)
        if self.edge_color is not None:
            edge.set_color(self.edge_color)
        if self.edge_fontcolor is not None:
            edge.set_fontcolor(self.edge_fontcolor)

        self.graph.add_edge(edge)

    def visit_Node(self, node, pydot_node=None):
        "Visit children and add edges to their Graphviz nodes."
        if pydot_node is None:
            pydot_node = self.create_node(node)

        nodes_dict = self.visitchildren(node)
        attrs = self.context.getchildren(node)

        for attr in attrs:
            values = nodes_dict.get(attr, None)
            if values is not None:
                if isinstance(values, list):
                    for value in values:
                        self.add_edge(pydot_node, value)
                else:
                    self.add_edge(pydot_node, values, attr)

        return pydot_node

    def visit_FunctionNode(self, node):
        "Create a graphviz graph"
        self.graph = pydot.Dot(self.name, graph_type='digraph')

        pydot_function = self.create_node(node)
        pydot_body = self.visit(node.body)

        # Create artificial arguments for brevity
        pydot_args = pydot.Node("Arguments (omitted)")
        self.add_node(pydot_args)

        self.add_edge(pydot_function, pydot_body)
        self.add_edge(pydot_function, pydot_args)

        return self.graph
