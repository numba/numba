# -*- coding: utf-8 -*-

"""
This module defines ordering schemes for virtual methods and attributes.

If we use hash-based virtual (method/attribute) tables, we don't care about
the ordering. If we're using a C++ like virtual method/attribute table (like
normal Python extension types do for attributes), we need to have a layout
compatible with base classes (i.e. we may only add more attributes, but not
reorder any existing ones).
"""

from __future__ import print_function, division, absolute_import

from numba.traits import traits, Delegate
from numba import error

#------------------------------------------------------------------------
# Virtual Tables
#------------------------------------------------------------------------

@traits
class AbstractTable(object):

    # Ordered attribute names
    attributes = None

    # Dict mapping attribute names to attribute entities
    attrdict = None

    py_class = Delegate('table')

    def __init__(self, table):
        self.table = table

    @property
    def parents(self):
        cls = type(self)
        return list(map(cls, self.table.parents))

@traits
class VTable(AbstractTable):

    attributes = Delegate('table', 'methodnames')
    attrdict = Delegate('table', 'methoddict')

@traits
class AttributeTable(AbstractTable):

    attributes = Delegate('table', 'attributes')
    attrdict = Delegate('table', 'attributedict')

#------------------------------------------------------------------------
# Table Entry Ordering (Virtual Method / Attribute Ordering)
#------------------------------------------------------------------------

def sort_parents(table):
    "Sort parent tables by size"
    return sorted(table.parents, key=lambda tab: len(tab.attrdict))

def unordered(table):
    "Return table entities in a random order"
    return list(table.attrdict)

def extending(table):
    """
    Order the table entities according to the given parent tables, i.e.
    we can only extend existing tables.
    """
    if not table.parents:
        return unordered(table)

    parents = sort_parents(table)
    biggest_table = parents[-1]

    appending_attributes = set(table.attrdict) - set(biggest_table.attributes)
    return biggest_table.attributes + list(appending_attributes)


# ______________________________________________________________________
# Validate Table Ordering

def validate_extending_order_compatibility(table):
    parents = sort_parents(table)
    tables = parents + [table]

    for table_smaller, table_bigger in zip(tables, tables[1:]):
        names1 = table_smaller.attributes
        names2 = table_bigger.attributes[:len(table_smaller.attributes)]

        if names1 != names2:
            raise error.NumbaError(
                "Cannot create compatible attribute or method ordering for "
                "base classes '%s' and '%s'" % (
                    table_smaller.py_class.__name__,
                    table_bigger.py_class.__name__))

