import ast

class FixMissingLocations(ast.NodeVisitor):
    """
    Fix missing source position information.
    """

    def __init__(self, lineno, col_offset, override=False):
        self.lineno = lineno
        self.col_offset = col_offset
        self.override = override

    def visit(self, node):
        super(FixMissingLocations, self).visit(node)
        if not hasattr(node, 'lineno') or self.override:
            node.lineno = self.lineno
            node.col_offset = self.col_offset
        else:
            self.lineno = node.lineno
            self.col_offset = node.col_offset