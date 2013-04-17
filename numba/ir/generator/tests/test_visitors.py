# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys

from .testutils import generate_in_memory
from .. import generator, visitorgen, astgen

#------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------

def load_testschema1():
    codegens = astgen.codegens + visitorgen.codegens
    file_allocator = generate_in_memory("testschema1.asdl", codegens)

    interface = generator.generate_module(file_allocator, "interface.py")
    sys.modules['interface'] = interface

    nodes = generator.generate_module(file_allocator, "nodes.py")
    sys.modules['nodes'] = nodes

    visitor = generator.generate_module(file_allocator, "visitor.py")
    transformer = generator.generate_module(file_allocator, "transformer.py")

    return nodes, visitor, transformer

nodes, visitor, transformer = load_testschema1()

#------------------------------------------------------------------------
# Visitor testers
#------------------------------------------------------------------------

class TestVisitor(visitor.Visitor):

    def visit_Bar(self, node):
        print("Bar:", node.e1, node.e2)
        self.generic_visit(node)

    def visit_SomeExpr(self, node):
        print("SomeExpr:", node)

class TestTransformer(transformer.Transformer):

    def visit_SomeExpr(self, node):
        return nodes.SomeOtherExpr(node.n)

#------------------------------------------------------------------------
# Test funcs
#------------------------------------------------------------------------

def test_visitor():
    """
    >>> test_visitor()
    Bar: Bar(e1=SomeExpr(n=10), e2=SomeExpr(n=11)) SomeExpr(n=12)
    Bar: SomeExpr(n=10) SomeExpr(n=11)
    SomeExpr: SomeExpr(n=10)
    SomeExpr: SomeExpr(n=11)
    SomeExpr: SomeExpr(n=12)
    """
    e1 = nodes.SomeExpr(10)
    e2 = nodes.SomeExpr(11)
    expr = nodes.Bar(nodes.Bar(e1, e2), nodes.SomeExpr(12))
    TestVisitor().visit(expr)

def test_transformer():
    """
    >>> test_transformer()
    Bar(e1=Bar(e1=SomeOtherExpr(n=10), e2=SomeOtherExpr(n=11)), e2=SomeOtherExpr(n=12))
    """
    e1 = nodes.SomeExpr(10)
    e2 = nodes.SomeExpr(11)
    expr = nodes.Bar(nodes.Bar(e1, e2), nodes.SomeExpr(12))
    result = TestTransformer().visit(expr)
    print(result)


if __name__ == '__main__':
    import doctest
    sys.exit(0 if doctest.testmod().failed == 0 else 1)