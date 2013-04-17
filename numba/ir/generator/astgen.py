# -*- coding: utf-8 -*-

"""
Generate Python AST nodes and Cython pxd files.

Note: This file needs to be easy to externalize (factor out into another
project).
"""

from __future__ import print_function, division, absolute_import

import os
from textwrap import dedent
from functools import partial

# from . import generator
from numba.ir.generator import generator

root = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(root, "tests")

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

py_preamble = """
import types

class TypedProperty(object):
    '''Defines a class property that does a type check in the setter.'''

    def __new__(cls, ty, doc, default=None):
        rv = super(TypedProperty, cls).__new__(cls)
        cls.__init__(rv, ty, doc, default)
        return property(rv.getter, rv.setter, rv.deleter, doc)

    def __init__(self, ty, doc, default=None):
        self.propname = '_property_%d' % (id(self),)
        self.default = default
        self.ty = ty
        self.doc = doc

    def getter(self, obj):
        return getattr(obj, self.propname, self.default)

    def setter(self, obj, new_val):
        if not isinstance(new_val, self.ty):
            raise ValueError(
                'Invalid property setting, expected instance of type(s) %r '
                '(got %r).' % (self.ty, type(new_val)))
        setattr(obj, self.propname, new_val)

    def deleter(self, obj):
        delattr(obj, self.propname)
"""

cy_preamble = """
cimport cython

cdef class Visitor(object):
    pass
"""

def format_field(field):
    if field.opt:
        type = "(%s, types.NoneType)" % field.type
    else:
        type = field.type

    format_dict = dict(name=field.name, type=type)
    return '%(name)s = TypedProperty(%(type)s, "%(name)s")' % format_dict

def format_stats(pattern, indent, stats):
    pattern = pattern + " " * indent
    return pattern.join(stats)

#------------------------------------------------------------------------
# Class Generation
#------------------------------------------------------------------------

class Class(object):
    def __init__(self, name, base, doc, fields=(),
                 attributes=()):
        self.name = name
        self.base = base
        self.doc = doc
        self.fields = fields
        self.attributes = attributes

class PyClass(Class):
    """
    Generate Python AST classes.
    """

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

class CyClass(Class):
    """
    Generate classes in pxd overlay.
    """

    def __str__(self):
        fields = ["cdef public %s %s" % (field.type, field.name)
                      for field in self.fields]
        fmtdict = {
            'name': self.name,
            'base': self.base,
            'fields': format_stats("\n", 4, fields)
        }

        return dedent('''
            cdef class %(name)s(%(base)s):
                %(fields)s
                cdef visit(self, Visitor visitor)
        ''') % fmtdict


#------------------------------------------------------------------------
# Code Generators
#------------------------------------------------------------------------

class ASTNodeCodeGen(generator.Codegen):
    """
    Generate Python AST nodes.
    """

    def __init__(self, out_filename, preamble, Class):
        super(ASTNodeCodeGen, self).__init__(out_filename)
        self.preamble = preamble
        self.Class = Class

    def generate(self, emitter, asdl_tree, schema):
        emitter.emit(self.preamble)
        emitter.emit(self.Class("AST", "object", doc="AST root node."))
        for rulename, rule in schema.dfns.iteritems():
            self.emit_rule(emitter, schema, rulename, rule)

    def emit_rule(self, emitter, schema, rulename, rule):
        "Emit code for a rule (a nonterminal)"
        emitter.emit(self.Class(rulename, "AST", doc=str(rule)))
        # print(rulename, rule.fields)
        if rule.is_sum:
            for subtype in rule.fields:
                self.emit_sum(emitter, schema, rulename, rule, subtype)

    def emit_sum(self, emitter, schema, rulename, rule, sumtype):
        fields = schema.types[sumtype]
        emitter.emit(self.Class(sumtype, rulename, doc=sumtype, fields=fields))

#------------------------------------------------------------------------
# Global Exports
#------------------------------------------------------------------------

codegens = [
    ASTNodeCodeGen("nodes.py", py_preamble, PyClass),
    ASTNodeCodeGen("nodes.pxd", cy_preamble, CyClass),
]


if __name__ == '__main__':
    schema_filename = os.path.join(testdir, "testschema1.asdl")
    generator.generate_from_file(schema_filename, codegens, root)
