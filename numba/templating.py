# -*- coding: utf-8 -*-
"""
String templating support. The following syntax is supported:

    1) Arbitrary Python code
    2) Placeholders for AST sub-tree substitutions:

        {{some_node}}

    3) Typed, untyped and code template variables:

        $some_var

        Typed and untyped variables are just artificial program
        (Python) variables.

            - Typed variables end up in the jit/autojit locals dict
            - Untyped variables are renameable/ssa variables
            - Code variables simply expand to arbitrary code
"""
from __future__ import print_function, division, absolute_import


import re
import ast
import string
import textwrap

import numba.decorators, numba.pipeline, numba.environment, numba.functions
from numba import *
from numba import nodes, symtab as symtab_module
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
        return '__numba_temp%d_%s' % (self.count(), name)

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
        self.code = code
        self.sep = "\n"
        if not temp_name:
            assert code
            self.codes = []

    def __str__(self):
        if self.temp_name:
            return self.temp_name

        return self.sep.join(self.codes) or "pass"

    @property
    def node(self):
        node = ast.Name(self.temp_name, ast.Load())
        node.type = self.type
        node.variable = self
        return node

class TemplateContext(object):
    """
    The context in which a template is evaluated. Allows defining
    template variables and a mechanism to merge those variables back
    into the function being compiled.
    """

    def __init__(self, context, template, env=None):
        # FIXME: Replace context with env.
        self.context = context
        self.templ = template
        self.variables = []
        self.nodes = {}
        self.env = env

        self.substituted_template = None

    def temp_var(self, name, type=None, code=False):
        "Create and add a new template $variable"
        var = TemplateVariable(name=name, type=type, is_local=True,
                               temp_name=not code and temp_name(name),
                               code=code)
        self.variables.append(var)
        return var

    def add_variable(self, var):
        "Add an external template $variable to this context"
        self.variables.append(var)

    def code_var(self, name):
        "Create a template $variable that expands to generated code statements"
        return self.temp_var(name, code=True)

    def temp_vars(self, *names):
        "Create a number of template $variables"
        for name in names:
            yield self.temp_var(name)

    def code_vars(self, *names):
        "Create a number of code $variables"
        for name in names:
            yield self.code_var(name)

    def string_substitute(self, s):
        if self.variables:
            d = dict((var.name, str(var)) for var in self.variables)
            s = string.Template(s).substitute(d)

        return s

    def get_vars_symtab(self):
        return dict((var.temp_name, Variable(name=var.temp_name,
                                             is_local=True, type=var.type))
                        for var in self.variables if not var.code)

    def update_locals(self, locals_dict):
        """
        Update an external jit/autojit locals dict with our
        template $variables
        """
        for var in self.variables:
            if not var.code and var.type is not None:
                assert var.name not in locals_dict
                locals_dict[var.temp_name] = var.type

    def template(self, substitutions):
        """
        Run a template and perform the given {{substitutions}}
        """
        s = textwrap.dedent(self.templ)
        s = self.string_substitute(s)

        # print s
        self.substituted_template = s

        # template_variables = dict((var.name, var) for var in self.variables)
        tree = template(s, substitutions, self.get_vars_symtab())
        return tree

    def template_type_infer(self, substitutions, **kwargs):
        tree = self.template(substitutions)

        symtab = kwargs.get('symtab', None)
        if self.variables or symtab:
            vars = self.get_vars_symtab()
            symtab = dict(symtab or {}, **vars)
            kwargs['symtab'] = symtab_module.Symtab(symtab)

        return dummy_type_infer(self.context, tree, env=self.env, **kwargs)

def dummy_type_infer(context, tree, order=['type_infer', 'type_set'], env=None,
                     **kwargs):
    assert env is not None
    func_obj = kwargs.pop('func')
    func_env = env.translation.crnt.inherit(func=func_obj, ast=tree,
                                            func_signature=void(),
                                            **kwargs)
    numba.pipeline.run_env(
        env, func_env, pipeline_name='dummy_type_infer',
        function_level=1, locals=func_env.locals, **kwargs)
    symtab = func_env.symtab
    ast = func_env.ast
    return symtab, ast

def template(s, substitutions, template_variables=None):
    s = textwrap.dedent(s)
    replaced = [0]
    def replace(ident):
        replaced[0] += 1
        return '%s%s' % (prefix, ident.group(1))

    source = re.sub('{{(.*?)}}', replace, s)
    tree = ast.parse(source)

    if replaced:
        tree = Interpolate(substitutions, template_variables).visit(tree)

    return ast.Suite(body=tree.body)

def template_simple(s, **substitutions):
    return template(s, substitutions)

class Interpolate(ast.NodeTransformer):
    """
    Interpolate template substitutions.

    substitutions: { var_name : substitution_node }

        This is for {{var_name}} template syntax. This allows
        substituting arbitrary asts is specific places.

    template_variables: [TemplateVariable(...)]

        This is for $var syntax. This allows the introductions of typed
        or untyped variables, as well as code variables.
    """

    def __init__(self, substitutions, template_variables):
        self.substitutions = substitutions
        self.template_variables = template_variables or {}
        self.make_substitutions_clonenodes()

    def make_substitutions_clonenodes(self):
        for name, replacement in self.substitutions.iteritems():
            is_clone = isinstance(replacement, (nodes.CloneableNode,
                                                nodes.CloneNode,
                                                ast.Name,   # atomic
                                                nodes.TempLoadNode))
            have_type = hasattr(replacement, 'type')
            if not is_clone and have_type:
                self.substitutions[name] = nodes.CloneableNode(replacement)

    def visit_Name(self, node):
        if node.id.startswith(prefix):
            name = node.id[len(prefix):]
            node = self.substitutions[name]
            if isinstance(node, nodes.CloneableNode):
                self.substitutions[name] = node.clone

        elif node.id in self.template_variables:
            node.variable = self.template_variables[node.id]
            node = nodes.MaybeUnusedNode(node)

        return node
