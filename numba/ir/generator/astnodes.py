# -*- coding: utf-8 -*-

"""
Generate Python AST nodes and Cython pxd files.
"""

from __future__ import print_function, division, absolute_import

import os
from textwrap import dedent
from functools import partial

from numba.ir.generator import generator

root = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(root, "tests")

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

preamble = """
import types
from numba.utils import TypedProperty
"""

def format_field(field):
    if field.opt:
        type = "(%s, types.NoneType)" % field.type
    else:
        type = field.type

    format_dict = dict(name=field.name, type=type)
    return '%(name)s = TypedProperty("%(name)s", %(type)s)' % format_dict

def format_stats(pattern, indent, stats):
    pattern = pattern + " " * indent
    return pattern.join(stats)

class Class(object):
    def __init__(self, name, base, doc, fields=(),
                 attributes=()):
        self.name = name
        self.base = base
        self.doc = doc
        self.fields = fields
        self.attributes = attributes

    def __str__(self):
        fieldnames = [str(field.name) for field in self.fields]
        fields = map(repr, fieldnames)
        properties = map(format_field, self.fields)
        attributes = map(repr, self.attributes)
        if fieldnames:
            initialize = ["self.%s = %s" % (name, name) for name in fieldnames]
        else:
            initialize = ["pass"]

        format_dict = dict(
            name=self.name, base=self.base, doc=self.doc,
            fields      = format_stats(",\n", 8, fields),
            attributes  = format_stats(",\n", 8, attributes),
            properties  = format_stats("\n", 4, properties),
            params      = ", ".join(fieldnames),
            initialize  = format_stats("\n", 8, initialize),
        )

        return dedent('''
            class %(name)s(%(base)s):
                """
                %(doc)s
                """

                _fields = (
                    %(fields)s
                )

                _attributes = (
                    %(attributes)s
                )

                # Properties
                %(properties)s

                def __init__(self, %(params)s):
                    %(initialize)s

                def visit(self, visitor):
                    return visitor.visit_%(name)s(self)

        ''') % format_dict

#------------------------------------------------------------------------
# Code Generators
#------------------------------------------------------------------------

class PythonASTNodeCodeGen(generator.Codegen):
    """
    Generate Python AST nodes.
    """

    def generate(self, emit_code, asdl_tree, schema):
        out = self.open_sourcefile("nodes.py")
        emit = partial(emit_code, out)

        emit(preamble)
        emit(Class("AST", "object", doc="AST root node."))
        for rulename, rule in schema.dfns.iteritems():
            self.emit_rule(emit, schema, rulename, rule)

    def emit_rule(self, emit, schema, rulename, rule):
        "Emit code for a rule (a nonterminal)"
        emit(Class(rulename, "AST", doc=str(rule)))
        # print(rulename, rule.fields)
        if rule.is_sum:
            for subtype in rule.fields:
                self.emit_sum(emit, schema, rulename, rule, subtype)

    def emit_sum(self, emit, schema, rulename, rule, sumtype):
        fields = schema.types[sumtype]
        emit(Class(sumtype, rulename, doc=sumtype, fields=fields))


if __name__ == '__main__':
    schema_file = os.path.join(testdir, "testschema1.asdl")
    generator.generate_from_file(schema_file, [PythonASTNodeCodeGen], root)
