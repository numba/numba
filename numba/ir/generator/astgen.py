# -*- coding: utf-8 -*-

"""
Generate Python AST nodes and Cython pxd files.

Note: This file needs to be easy to externalize (factor out into another
project).
"""

from __future__ import print_function, division, absolute_import

import os
import ast
from textwrap import dedent
from functools import partial

from . import generator
from . import naming
from .classgen import Class, ClassCodegen
from .formatting import py_formatter, cy_formatter

root = os.path.dirname(os.path.abspath(__file__))
testdir = os.path.join(root, "tests")

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

py_preamble = """
import cython

# ASDL builtin types
# identifier = str
# string = basestring

class TypedProperty(object):
    '''Defines a class property that does a type check in the setter.'''

    def __new__(cls, ty, doc, default=None):
        rv = super(TypedProperty, cls).__new__(cls)
        cls.__init__(rv, ty, doc, default)
        return property(rv.getter, rv.setter, rv.deleter, doc)

    def __init__(self, ty, typename, attrname, default=None):
        self.propname = '_property_%d' % (id(self),)
        self.default = default
        if not isinstance(ty, tuple):
            ty = (ty,)
        self.ty = ty
        self.typename = typename
        self.attrname = attrname

    def getter(self, obj):
        return getattr(obj, self.propname, self.default)

    def setter(self, obj, new_val):
        if not isinstance(new_val, self.ty):
            ty = tuple(t.__name__ for t in self.ty)
            raise ValueError(
                "Invalid type for attribute '%s.%s', "
                "expected instance of type(s) %r "
                "(got %r)." % (self.typename, self.attrname,
                               ty, type(new_val).__name__))
        setattr(obj, self.propname, new_val)

    def deleter(self, obj):
        delattr(obj, self.propname)
"""

cy_preamble = """
cimport cython
from %s cimport GenericVisitor

# ctypedef str identifier
# ctypedef str string
# ctypedef bint bool
""" % (naming.interface,)

def format_field(classname, field):
    type = py_formatter.format_type(field.type)
    if field.opt:
        # types.NoneType is not in Python 3, type(None) should work
        # for both Python 2 and 3.
        type = "(%s, type(None))" % (type,)

    format_dict = dict(name=field.name, type=type, classname=classname)
    return ('%(classname)s.%(name)s = TypedProperty(%(type)s, '
                '"%(classname)s", "%(name)s")' % format_dict)

#------------------------------------------------------------------------
# Class Formatting
#------------------------------------------------------------------------

class PyClass(Class):
    """
    Generate Python AST classes.
    """

    def __str__(self):
        fieldnames = [str(field.name) for field in self.fields]
        fields = map(repr, fieldnames)
        properties = [format_field(self.name, field)
                          for field in self.fields] or ["pass"]

        if self.attributes is not None:
            attributes = map(repr, self.attributes)
            attributes = py_formatter.format_stats(",\n", 8, attributes)
            attributes = "_attributes = (" + attributes + ")"
        else:
            attributes = "# inherit _attributes"

        if fieldnames:
            initialize = ["self.%s = %s" % (name, name) for name in fieldnames]
        else:
            initialize = ["pass"]

        fmtstring = ", ".join("%s=%%s" % name for name in fieldnames)
        fmtargs = "(%s)" % ", ".join(py_formatter.get_fields(self.fields))

        format_dict = dict(
            name=self.name, base=self.base, doc=self.doc,
            fields      = py_formatter.format_stats(",\n", 8, fields),
            attributes  = attributes,
            properties  = py_formatter.format_stats("\n", 4, properties),
            params      = ", ".join(fieldnames),
            initialize  = py_formatter.format_stats("\n", 8, initialize),
            fmtstring   = fmtstring,
            fmtargs     = fmtargs,
        )

        return dedent('''
            class %(name)s(%(base)s):
                """
                %(doc)s
                """

                _fields = (
                    %(fields)s
                )

                %(attributes)s

                def __init__(self, %(params)s):
                    %(initialize)s

                def accept(self, visitor):
                    return visitor.visit_%(name)s(self)

                def __repr__(self):
                    return "%(name)s(%(fmtstring)s)" %% %(fmtargs)s

            if not cython.compiled:
                # Properties
                %(properties)s

        ''') % format_dict

class CyClass(Class):
    """
    Generate classes in pxd overlay.
    """

    def __str__(self):
        f = cy_formatter
        fields = ["cdef public %s %s" % (f.format_type(field.type), field.name)
                      for field in self.fields]
        fmtdict = {
            'name': self.name,
            'base': self.base,
            'fields': cy_formatter.format_stats("\n", 4, fields)
        }

        return dedent('''
            cdef class %(name)s(%(base)s):
                %(fields)s
                cpdef accept(self, GenericVisitor visitor)
        ''') % fmtdict

#------------------------------------------------------------------------
# Global Exports
#------------------------------------------------------------------------

def make_root_class(Class):
    return Class("AST", "object", doc="AST root node.", attributes=())

codegens = [
    ClassCodegen(naming.nodes + '.py', py_preamble,
                 PyClass, make_root_class(PyClass)),
    ClassCodegen(naming.nodes + '.pxd', cy_preamble,
                 CyClass, make_root_class(CyClass)),
]


if __name__ == '__main__':
    schema_filename = os.path.join(testdir, "testschema1.asdl")
    generator.generate_from_file(schema_filename, codegens, root)
