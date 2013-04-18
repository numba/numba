# -*- coding: utf-8 -*-

"""
Generate Python visitors and Cython pxd files.
"""

from __future__ import print_function, division, absolute_import

import os

from . import generator
from . import naming
from .formatting import py_formatter

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

interface_class = '''

def iter_fields(node):
    """
    Yield a tuple of ``(fieldname, value)`` for each field in ``node._fields``
    that is present on *node*.
    """
    result = []
    for field in node._fields:
        try:
            result.append((field, getattr(node, field)))
        except AttributeError:
            pass

    return result

class GenericVisitor(object):

    def visit(self, node):
        return node.accept(self)

    def generic_visit(self, node):
        """Called explicitly by the user from an overridden visitor method"""
        raise NotImplementedError

'''

pxd_interface_class = """\
cimport %s

cdef class GenericVisitor(object):
    cpdef generic_visit(self, node)
""" % (naming.nodes,)


# TODO: We can also make 'visitchildren' dispatch quickly

visitor_class = '''
from %s import GenericVisitor, iter_fields
from %s import AST

__all__ = ['Visitor']

class Visitor(GenericVisitor):

    def generic_visit(self, node):
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        item.accept(self)
            elif isinstance(value, AST):
                value.accept(self)

''' % (naming.interface, naming.nodes)

transformer_class = """
from %s import GenericVisitor, iter_fields
from %s import AST

__all__ = ['Transformer']

class Transformer(GenericVisitor):

    def generic_visit(self, node):
        for field, old_value in iter_fields(node):
            old_value = getattr(node, field, None)
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, AST):
                        value = value.accept(self)
                        if value is None:
                            continue
                        elif not isinstance(value, AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, AST):
                new_node = old_value.accept(self)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

""" % (naming.interface, naming.nodes,)

pxd_visitor_class = """
from %s cimport GenericVisitor

cdef class Visitor(GenericVisitor):
    pass
""" % (naming.interface,)

pxd_transformer_class = """
from %s cimport GenericVisitor

cdef class Transformer(GenericVisitor):
    pass
""" % (naming.interface,)

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

def make_visit_stats(schema, fields, inplace):
    stats = []

    # ["self.attr_x"]
    field_accessors = py_formatter.get_fields(fields, obj="node")

    for field, field_access in zip(fields, field_accessors):
        field_type = str(field.type)
        if field_type not in schema.dfns and field_type not in schema.types:
            # Not an AST node
            continue

        s = "%s.accept(self)" % field_access

        if inplace:
            # Mutate in-place (transform)
            s = "%s = %s" % (field_access, s)

        if field.opt:
            # Guard for None
            s = "if %s is not None: %s" % (field_access, s)

        stats.append(s)

    if inplace:
        stats.append("return node")

    return stats or ["pass"]

#------------------------------------------------------------------------
# Method Generation
#------------------------------------------------------------------------

class Method(object):
    def __init__(self, schema, name, fields):
        self.schema = schema
        self.name = name
        self.fields = fields

class InterfaceMethod(Method):
    def __str__(self):
        return (
           "    def visit_%s(self, node):\n"
           "        raise NotImplementedError\n"
           "\n"
       ) % (self.name,)

class PyMethod(Method):

    inplace = None

    def __str__(self):
        stats = make_visit_stats(self.schema, self.fields, self.inplace)
        return (
           "    def visit_%s(self, node):\n"
           "        %s\n"
           "\n"
       ) % (self.name, py_formatter.format_stats("\n", 8, stats))

class PyVisitMethod(PyMethod):
    inplace = False

class PyTransformMethod(PyVisitMethod):
    inplace = True

class PxdMethod(Method):
    def __str__(self):
        return "    cpdef visit_%s(self, %s.%s node)\n" % (self.name,
                                                           naming.nodes,
                                                           self.name)

#------------------------------------------------------------------------
# Code Generators
#------------------------------------------------------------------------

class VisitorCodegen(generator.SimpleCodegen):
    """
    Generate Python AST nodes.
    """

    def __init__(self, out_filename, preamble, Method):
        super(VisitorCodegen, self).__init__(out_filename)
        self.preamble = preamble
        self.Method = Method

    def emit_preamble(self, emitter, schema):
        emitter.emit(self.preamble)

    def emit_sum(self, emitter, schema, rulename, rule, sumtype):
        fields = schema.types[sumtype]
        emitter.emit(self.Method(schema, sumtype, fields))

#------------------------------------------------------------------------
# Global Exports
#------------------------------------------------------------------------

codegens = [
    VisitorCodegen(naming.interface + '.py', interface_class, InterfaceMethod),
    VisitorCodegen(naming.interface + '.pxd', pxd_interface_class, PxdMethod),
    VisitorCodegen(naming.visitor + '.py', visitor_class, PyVisitMethod),
    generator.UtilityCodegen(naming.visitor + '.pxd', pxd_visitor_class),
    VisitorCodegen(naming.transformer + '.py', transformer_class, PyTransformMethod),
    generator.UtilityCodegen(naming.transformer + '.pxd', pxd_transformer_class),
]

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    testdir = os.path.join(root, "tests")
    schema_filename = os.path.join(testdir, "testschema1.asdl")
    generator.generate_from_file(schema_filename, codegens, root)