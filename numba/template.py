import re
import ast
import string
import types
import ctypes
import textwrap

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

    def temp_name(self, name):
        return '__numba_%s_temp%d' % (name, self.count())

_temp_name = TempName()

def temp_name(name=''):
    return _temp_name.temp_name(name)

class TemplateVariable(Variable):
    """
    A fake variable used in a template. The 'code' is substituted using string
    interpolation before the source is parsed. If the type is set, the symbol
    table is updated. The default 'code' is a temporary variable name.
    """

    def __init__(self, type, name, temp_name=None, code=False, **kwargs):
        super(TemplateVariable, self).__init__(type, name=name, **kwargs)
        self.temp_name = temp_name
        if not temp_name:
            assert code
            self.codes = []

    def __str__(self):
        if self.temp_name:
            return self.temp_name

        return "\n".join(self.codes)

    @property
    def node(self):
        node = ast.Name(self.temp_name, ast.Load())
        node.type = self.type
        node.variable = self
        return node

class TemplateContext(object):

    def __init__(self, context, template):
        self.context = context
        self.template = template
        self.variables = []
        self.nodes = {}

        self.substituted_template = None

    def temp_var(self, name, type=None, code=False):
        var = TemplateVariable(name=name, type=type, is_local=True,
                               temp_name=not code and temp_name(name),
                               code=code)
        self.variables.append(var)
        return var

    def code_var(self, name):
        return self.temp_var(name, code=True)

    def temp_vars(self, *names):
        for name in names:
            yield self.temp_var(name)

    def code_vars(self, *names):
        for name in names:
            yield self.code_var(name)

    def string_substitute(self, s):
        if self.variables:
            d = dict((var.name, str(var)) for var in self.variables)
            s = string.Template(s).substitute(d)

        return s

    def template_type_infer(self, substitutions, **kwargs):
        s = textwrap.dedent(self.template)
        s = self.string_substitute(s)
        # print s
        self.substituted_template = s

        symtab = kwargs.get('symtab', None)
        if self.variables or symtab:
            vars = dict((var.name, Variable(name=var.name, is_local=True,
                                            type=var.type))
                            for var in self.variables if var.type)
            kwargs['symtab'] = dict(symtab or {}, **vars)

        tree = template(s, substitutions)
        return dummy_type_infer(self.context, tree, **kwargs)


def dummy_type_infer(context, tree, order=['type_infer', 'type_set'], **kwargs):
    def dummy():
        pass
    return numba.pipeline.run_pipeline(context, dummy, tree, void(),
                                       order=order, **kwargs)

def template(s, substitutions):
    s = textwrap.dedent(s)
    replaced = [0]
    def replace(ident):
        replaced[0] += 1
        return '%s%s' % (prefix, ident.group(1))

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