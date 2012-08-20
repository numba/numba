# A handy translator that converts control flow into the appropriate
# llvm_cbuilder constructs
import inspect, functools, ast
import pprint
def translate(func):
    # TODO use meta package
    wrapper = functools.wraps(func)
    source = inspect.getsource(func)
    lines = list(source.splitlines())

     # skip first two lines
    first_line_len = len(lines[2])
    indent = first_line_len - len(lines[2].lstrip())
    source = '\n'.join(line[indent:] for line in lines[2:])

    caller_frame = inspect.currentframe().f_back

    tree = ast.parse(source)
    tree = ExpandControlFlow().visit(tree)
    tree = ast.fix_missing_locations(tree)

    # prepare locals for execution
    local_dict = locals()
    local_dict.update(caller_frame.f_locals)
    local_dict.update(caller_frame.f_globals)


    return eval(compile(tree, '<string>', 'exec'))

_if_else_template = '''
with self.ifelse(__CONDITION__) as _ifelse_:
    with _ifelse_.then():
        __THEN__
    with _ifelse_.otherwise():
        __OTHERWISE__
'''

_return_template = 'self.ret(__RETURN__)'

def load_template(string):
    tree = ast.parse(string)  # return a Module
    return tree.body[0]       # get the first item of body

class ExpandControlFlow(ast.NodeTransformer):
    def visit_If(self, node):
        condition = node.test
        mapping = {
            '__CONDITION__' : node.test,
            '__THEN__'      : node.body,
            '__OTHERWISE__' : node.orelse,
        }

        ifelse = load_template(_if_else_template)
        ifelse = NameReplacer(mapping).visit(ifelse)
        newnode = ast.copy_location(ifelse, node)
        return self.generic_visit(newnode)

    def visit_Return(self, node):
        mapping = {'__RETURN__' : node.value}
        ret = load_template(_return_template)
        repl = NameReplacer(mapping).visit(ret)
        return ast.copy_location(repl, node)

class NameReplacer(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_With(self, node):
        if (len(node.body)==1
          and isinstance(node.body[0], ast.Expr)
          and isinstance(node.body[0].value, ast.Name)):
            try:
                repl = self.mapping.pop(node.body[0].value.id)
            except KeyError:
                pass
            else:
                old = node.body[0]
                node.body = repl
        return self.generic_visit(node)

    def visit_Name(self, node):
        if type(node.ctx) is ast.Load:
            try:
                repl = self.mapping.pop(node.id)
            except KeyError:
                pass
            else:
                if repl is not None and not isinstance(repl, list):
                    return ast.copy_location(repl, node)
        return node

