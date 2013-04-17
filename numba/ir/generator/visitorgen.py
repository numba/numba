# -*- coding: utf-8 -*-

"""
Generate Python visitors and Cython pxd files.
"""

from __future__ import print_function, division, absolute_import

import os

from . import generator
from .formatting import format_stats, get_fields

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

# TODO: We can also make 'visitchildren' dispatch quickly

interface_class = """
class GenericVisitor(object):

    def generic_visit(self, node):
        typename = type(node).__name__
        m = getattr(self, typename)
        m(self, node)

"""

pxd_interface_class = """\
from nodes cimport *

cdef class GenericVisitor(object):
    cpdef generic_visit(self, node)
"""

visitor_class = """
from interface import GenericVisitor

class Visitor(GenericVisitor):
"""

transformer_class = """
from interface import GenericVisitor

class Transformer(GenericVisitor):
"""

pxd_visitor_class = """
from interface import GenericVisitor

cdef class Visitor(GenericVisitor):
    pass
"""

pxd_transformer_class = """
from interface import GenericVisitor

cdef class Transformer(GenericVisitor):
    pass
"""

#------------------------------------------------------------------------
# Code Formatting
#------------------------------------------------------------------------

def make_visit_stats(fields, inplace):
    stats = []
    for field, field_access in zip(fields, get_fields(fields, obj="node")):
        s = "%s.accept(self)" % field_access

        if inplace:
            # Mutate in-place (transform)
            s = "%s = %s" % (field_access, s)

        if field.opt:
            # Guard for None
            s = "if %s is not None: %s" % (field_access, s)

        stats.append(s)

    return stats

#------------------------------------------------------------------------
# Method Generation
#------------------------------------------------------------------------

class Method(object):
    def __init__(self, name, fields):
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
        stats = make_visit_stats(self.fields, self.inplace)
        return (
           "    def visit_%s(self, node):\n"
           "        %s\n"
           "\n"
       ) % (self.name, format_stats("\n", 8, stats))

class PyVisitMethod(PyMethod):
    inplace = False

class PyTransformMethod(PyVisitMethod):
    inplace = True

class PxdMethod(Method):
    def __str__(self):
        return "    cpdef visit_%s(self, %s node)\n" % (self.name, self.name)

#------------------------------------------------------------------------
# Code Generators
#------------------------------------------------------------------------

class VisitorCodeGen(generator.Codegen):
    """
    Generate Python AST nodes.
    """

    def __init__(self, out_filename, preamble, Method):
        super(VisitorCodeGen, self).__init__(out_filename)
        self.preamble = preamble
        self.Method = Method

    def generate(self, emitter, asdl_tree, schema):
        emitter.emit(self.preamble)
        for rulename, rule in schema.dfns.iteritems():
            self.emit_rule(emitter, schema, rulename, rule)

    def emit_rule(self, emitter, schema, rulename, rule):
        "Emit code for a rule (a nonterminal)"
        if rule.is_sum:
            for sumtype in rule.fields:
                self.emit_sum(emitter, schema, sumtype)

    def emit_sum(self, emitter, schema, sumtype):
        fields = schema.types[sumtype]
        emitter.emit(self.Method(sumtype, fields))

#------------------------------------------------------------------------
# Global Exports
#------------------------------------------------------------------------

codegens = [
    VisitorCodeGen("interface.py", interface_class, InterfaceMethod),
    VisitorCodeGen("interface.pxd", pxd_interface_class, PxdMethod),
    VisitorCodeGen("visitor.py", visitor_class, PyVisitMethod),
    generator.UtilityCodeGen("visitor.pxd", pxd_visitor_class),
    VisitorCodeGen("transformer.py", transformer_class, PyTransformMethod),
    generator.UtilityCodeGen("transformer.pxd", pxd_transformer_class),
]

if __name__ == '__main__':
    root = os.path.dirname(os.path.abspath(__file__))
    testdir = os.path.join(root, "tests")
    schema_filename = os.path.join(testdir, "testschema1.asdl")
    generator.generate_from_file(schema_filename, codegens, root)