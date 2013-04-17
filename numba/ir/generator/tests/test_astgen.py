# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import types

from .testutils import generate_in_memory, generate_module
from .. import generator, astgen

def load_testschema1():
    file_allocator = generate_in_memory("testschema1.asdl", astgen.codegens)
    m = generate_module(file_allocator, "nodes.py")
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
    assert issubclass(m.Bar, m.root)

    assert issubclass(m.SomeExpr, m.expr)

def test_valid_node_instantiation():
    """
    >>> test_valid_node_instantiation()
    """
    m = load_testschema1()

    # Valid
    e1 = m.SomeExpr(10)
    e2 = m.SomeExpr(11)

    result = m.Foo(m.Bar(e1, e2))
    # print(result)

    result = m.Bar(e1, None)
    # print(result)

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
    ValueError: Invalid property setting, expected instance of type(s) <class 'expr'> (got <type 'NoneType'>).
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod()