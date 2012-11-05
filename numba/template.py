import ast
import types
import ctypes

import numba.decorators, numba.pipeline, numba.functions
from numba import *
from numba import error, visitors, nodes
from numba.minivect import  minitypes
from numba import  _numba_types as numba_types
from numba.symtab import Variable

import logging
logger = logging.getLogger(__name__)

prefix = '__numba_template_'

class TempName(object):

    _count = 0

    def count(self):
        self._count += 1
        return self._count

    def __str__(self, name):
        return '__numba_%s_temp%d_' % (name, self.count())

_temp_name = TempName()

class TemplateVariable(Variable):
    """
    A fake variable used in a template. The 'code' is substituted using string
    interpolation before the source is parsed. If the type is set, the symbol
    table is updated. The default 'code' is a temporary variable name.
    """

    def __init__(self, type, name, code, codelist, **kwargs):
        super(TemplateVariable, self).__init__(type, name=name, **kwargs)
        self.code = code
        self.codes = []

    def __str__(self):
        if self.codes:
            return "\n".join(self.codes)

        return self.code

    @property
    def temp_name(self):
        assert not self.codes
        return self.code

    @property
    def node(self):
        return ast.Name(self.temp_name, ast.Load())

def temp_name(name=''):
    return _temp_name.temp_name(name)

def temp_var(name, type=None, code=None):
    return TemplateVariable(name, type, is_local=True,
                            code=code or temp_name(name))

def dummy_type_infer(context, tree, order=['type_infer', 'type_set'], **kwargs):
    def dummy():
        pass
    return numba.pipeline.run_pipeline(context, dummy, tree, void(),
                                       order=order, **kwargs)

def template_type_infer(context, s, substitutions, variables, **kwargs):
    symtab = kwargs.get('symtab', None)
    if variables or symtab:
        variables = variables or {}
        vars = [(name, Variable(var.name, is_local=True, type=var.type))
                    for var in variables if var.type]
        kwargs['symtab'] = dict(symtab or {}, **vars)

    if variables:
        d = dict((var.name, var.code) for var in variables)
        s = s % d

    tree = template(s, substitutions)
    return dummy_type_infer(context, tree, **kwargs)

def template(s, substitutions):
    replaced = [0]
    def replace(ident):
        replaced[0] += 1
        return '%s_%s' % (prefix, ident.group(1))

    source = re.sub('{{(.*?)}}', replace, s)
    tree = ast.parse(source)

    if replaced:
        tree = Interpolate(substitutions).visit(tree)

    return tree

def template_simple(s, **substitutions):
    return template(s, substitutions)

class Interpolate(ast.NodeTransformer):

    def __init__(self, substitutions):
        self.substitutions = substitutions

    def visit_Name(self, node):
        if node.id.startswith(prefix):
            return self.substitutions[node.id[len(prefix):]]
        return node