# -*- coding: utf-8 -*-

"""
Extension attribute table type. Supports ordered (struct) fields, or
unordered (hash-based) fields.
"""

import numba
from numba.typesystem import NumbaType, is_obj
from numba.typesystem.exttypes import ordering

#------------------------------------------------------------------------
# Extension Attributes Type
#------------------------------------------------------------------------

class ExtensionAttributesTableType(NumbaType):
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
