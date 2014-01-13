# -*- coding: utf-8 -*-

"""
Extension attribute table type. Supports ordered (struct) fields, or
unordered (hash-based) fields.
"""

from __future__ import print_function, division, absolute_import

import numba
from numba.typesystem import is_obj
from numba.exttypes import ordering

#------------------------------------------------------------------------
# Extension Attributes Type
#------------------------------------------------------------------------

class AttributeTable(object):
    """
    Type for extension type attributes.
    """

    def __init__(self, py_class, parents):
        self.py_class = py_class

        # List of parent extension attribute table types
        self.parents = parents

        # attribute_name -> attribute_type
        self.attributedict = {}

        # Ordered list of attribute names
        self.attributes = None

        # Set of inherited attribute names
        self.inherited = set()

    def to_struct(self):
        return numba.struct([(attr, self.attributedict[attr])
                                 for attr in self.attributes])

    def create_attribute_ordering(self, orderer=ordering.unordered):
        """
        Create a consistent attribute ordering with the base types.

            ordering ∈ { unordered, extending, ... }
        """
        self.attributes = orderer(ordering.AttributeTable(self))

    def need_tp_dealloc(self):
        """
        Returns whether this extension type needs a tp_dealloc, tp_traverse
        and tp_clear filled out.
        """
        if self.parent_type is not None and self.parent_type.need_tp_dealloc:
            result = False
        else:
            field_types = self.attribute_struct.fielddict.itervalues()
            result = any(map(is_obj, field_types))

        self._need_tp_dealloc = result
        return result

    def strtable(self):
        if self.attributes is None:
            return str(self.attributedict)

        return "{%s}" % ", ".join("%r: %r" % (name, self.attributedict[name])
                                      for name in self.attributes)

    def __repr__(self):
        return "AttributeTable(%s)" % self.strtable()

    @classmethod
    def empty(cls, py_class):
        table = AttributeTable(py_class, [])
        table.create_attribute_ordering()
        return table

    @classmethod
    def from_list(cls, py_class, attributes):
        "Create a final attribute table from a list of attribute (name, type)."
        table = AttributeTable(py_class, [])
        table.attributedict.update(attributes)
        table.attributes = [name for name, type in attributes]
        return table