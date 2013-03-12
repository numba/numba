import ast
import pprint

def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


class AST3to2(ast.NodeTransformer):

    def _visit_list(self, alist):
        new_values = []
        for value in alist:
            if isinstance(value, ast.AST):
                value = self.visit(value)
                if value is None:
                    continue
                elif not isinstance(value, ast.AST):
                    new_values.extend(value)
                    continue
            new_values.append(value)
        return new_values

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                old_value[:] = self._visit_list(old_value)
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def __visit_FunctionDef(self, node):
        new_node = ast.FunctionDef(args=self.visit_arguments(node.args),
                                   body=self._visit_list(node.body),
                                   decorator_list=self._visit_list(node.decorator_list),
                                   name=node.name)
        ast.copy_location(new_node, node)
        return new_node

    def visit_Index(self, node):
        if isinstance(node.value, ast.Ellipsis):
            return node.value
        return node

    def visit_arguments(self, node):
        ret = []
        for arg_node in node.args:
            if isinstance(arg_node, ast.arg):
                new_node = ast.Name(ctx=ast.Param(), id=arg_node.arg)
                ret.append(new_node)
            elif isinstance(arg_node, ast.Name):
                ret.append(arg_node)
            else:
                raise TypeError('Cannot transform node %r' % arg_node)
        return ast.arguments(args=ret, defaults=node.defaults,
                             kwarg=node.kwarg, vararg=node.vararg)
