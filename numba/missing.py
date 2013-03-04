import numba.visitors

class FixMissingLocations(numba.visitors.NumbaVisitor):
    """
    Fix missing source position information.
    """

    def __init__(self, context, func, ast, *args, **kwargs):
        super(FixMissingLocations, self).__init__(context, func, ast,
                                                  *args, **kwargs)
        self.lineno = getattr(ast, 'lineno', 1)
        self.col_offset = getattr(ast, 'col_offset', 0)

    def visit(self, node):
        if not hasattr(node, 'lineno'):
            node.lineno = self.lineno
            node.col_offset = self.col_offset

        super(FixMissingLocations, self).visit(node)


