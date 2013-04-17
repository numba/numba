# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import sys

from .testutils import generate_in_memory
from .. import generator, astgen, naming

def load_testschema1():
    file_allocator = generate_in_memory("testschema1.asdl", astgen.codegens)
    m = generator.generate_module(file_allocator, naming.nodes + '.py')
    return m

def test_ast_generation():
    """
    >>> test_ast_generation()
    """
    m = load_testschema1()

    # Nonterminals
    assert issubclass(m.root, m.AST)
    assert issubclass(m.expr, m.AST)

    # Terminals
    assert issubclass(m.Ham, m.root)
    assert issubclass(m.Foo, m.root)

    assert issubclass(m.SomeExpr, m.expr)
    assert issubclass(m.Bar, m.expr)

def test_valid_node_instantiation():
    """
    >>> test_valid_node_instantiation()
    Foo(leaf=Bar(e1=SomeExpr(n=10), e2=SomeExpr(n=11)))
    Bar(e1=SomeExpr(n=10), e2=None)
    """
    m = load_testschema1()

    # Valid
    e1 = m.SomeExpr(10)
    e2 = m.SomeExpr(11)

    result = m.Foo(m.Bar(e1, e2))
    print(result)

    result = m.Bar(e1, None)
    print(result)

def test_invalid_node_instantiation():
    """
    >>> m = load_testschema1()
    >>> e2 = m.SomeExpr(10)
    >>> m.Foo()
    Traceback (most recent call last):
      ...
    TypeError: __init__() takes exactly 2 arguments (1 given)
    >>> m.Bar(None, e2)
    Traceback (most recent call last):
      ...
    ValueError: Invalid type for attribute 'Bar.e1', expected instance of type(s) ('expr',) (got 'NoneType').
    """


if __name__ == '__main__':
    import doctest
    sys.exit(0 if doctest.testmod().failed == 0 else 1)
