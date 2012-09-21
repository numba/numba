# A handy translator that converts control flow into the appropriate
# llvm_cbuilder constructs
from numba.functions import _get_ast
import inspect, functools, ast
import logging

logger = logging.getLogger(__name__)

def translate(func):
    # TODO use meta package
    wrapper = functools.wraps(func)
    caller_frame = inspect.currentframe().f_back
    tree = _get_ast(func)
    tree = ast.copy_location(ast.Module(body=tree.body), tree)
    tree = ExpandControlFlow().visit(tree)
    tree = ast.fix_missing_locations(tree)

    # prepare locals for execution
    local_dict = locals()
    local_dict.update(caller_frame.f_locals)
    local_dict.update(caller_frame.f_globals)

    try:
        return eval(compile(tree, '<string>', 'exec'))
    except Exception, e:
        logger.debug(ast.dump(tree))
        from ArminRonacher import codegen # uses Armin Ronacher's codegen to debug
        # http://dev.pocoo.org/hg/sandbox/file/852a1248c8eb/ast/codegen.py
        logger.debug(codegen.to_source(tree))
        raise


_if_else_template = '''
with self.ifelse(__CONDITION__) as _ifelse_:
    with _ifelse_.then():
        __THEN__
    with _ifelse_.otherwise():
        __OTHERWISE__
'''

_while_template = '''
with self.loop() as _loop_:
    with _loop_.condition() as _setcond_:
        _setcond_(__CONDITION__)
    with _loop_.body():
        __BODY__
'''

_for_range_template = '''
with self.for_range(*__ARGS__) as (_loop_, __ITER__):
    __BODY__
'''

_return_template = 'self.ret(__RETURN__)'

_const_int_template = 'self.constant(C.int, __VALUE__)'
_const_long_template = 'self.constant(C.long, __VALUE__)'
_const_float_template = 'self.constant(C.double, __VALUE__)'

def load_template(string):
    '''
    Since ast.parse() returns a ast.Module node,
    it is more useful to trim the Module and get to the first item of body
    '''
    tree = ast.parse(string)  # return a Module
    assert isinstance(tree, ast.Module)
    return tree.body[0]       # get the first item of body

class ExpandControlFlow(ast.NodeTransformer):
    '''
    Expand control flow contructs.
    These are the most tedious thing to do in llvm_cbuilder.
    '''

    ## Use breadcumb to track parent nodes
    #    def __init__(self):
    #        self.breadcumb = []
    #
    #    def visit(self, node):
    #        self.breadcumb.append(node)
    #        try:
    #            return super(ExpandControlFlow, self).visit(node)
    #        finally:
    #            self.breadcumb.pop()
    #
    #    @property
    #    def parent(self):
    #        return self.breadcumb[-2]

    def visit_If(self, node):
        mapping = {
            '__CONDITION__' : node.test,
            '__THEN__'      : node.body,
            '__OTHERWISE__' : node.orelse,
        }

        ifelse = load_template(_if_else_template)
        ifelse = MacroExpander(mapping).visit(ifelse)
        newnode = self.generic_visit(ifelse)
        return ast.copy_location(newnode, node)

    def visit_While(self, node):
        mapping = {
            '__CONDITION__' : node.test,
            '__BODY__'      : node.body,
        }
        whileloop = load_template(_while_template)
        whileloop = MacroExpander(mapping).visit(whileloop)
        newnode = self.generic_visit(whileloop)
        return ast.copy_location(newnode, node)

    def visit_For(self, node):
        try:
            if node.iter.func.id not in ['range', 'xrange']:
                return node
        except AttributeError:
            return node

        mapping = {
            '__ITER__' : node.target,
            '__BODY__' : node.body,
            '__ARGS__' : ast.Tuple(elts=node.iter.args, ctx=ast.Load()),
        }

        forloop = load_template(_for_range_template)
        forloop = MacroExpander(mapping).visit(forloop)
        newnode = self.generic_visit(forloop)
        return ast.copy_location(newnode, node)

    def visit_Return(self, node):
        mapping = {'__RETURN__' : node.value}
        ret = load_template(_return_template)
        repl = MacroExpander(mapping).visit(ret)
        return ast.copy_location(repl, node)

    def visit_Num(self, node):
        '''convert immediate values
        '''
        typemap = {
            int   : _const_int_template,
            long  : _const_long_template,  # TODO: disable long for py3
            float : _const_float_template,
        }

        template = load_template(typemap[type(node.n)])

        mapping = {
            '__VALUE__' : node,
        }
        constant = MacroExpander(mapping).visit(template).value
        newnode = ast.copy_location(constant, node)
        return newnode

class MacroExpander(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_With(self, node):
        '''
        Expand X in the following:
            with blah:
                X
        Nothing should go before or after X.
        X must be a list of nodes.
        '''
        if (len(node.body)==1 # the body of
          and isinstance(node.body[0], ast.Expr)
          and isinstance(node.body[0].value, ast.Name)):
            try:
                repl = self.mapping.pop(node.body[0].value.id)
            except KeyError:
                pass
            else:
                old = node.body[0]
                node.body = repl

        return self.generic_visit(node) # recursively apply expand all macros

    def visit_Name(self, node):
        '''
        Expand all Name node to simple value
        '''

        try:
            repl = self.mapping.pop(node.id)
        except KeyError:
            pass
        else:
            if repl is not None and not isinstance(repl, list):
                return ast.copy_location(repl, node)
        return node

