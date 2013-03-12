from numba.typesystem import *

#------------------------------------------------------------------------
# Extension Attributes Type
#------------------------------------------------------------------------

class ExtensionAttributesTableType(NumbaType):
    """
    Type for extension type attributes.
    """

    def __init__(self, parents):
        # List of parent extension attribute table types
        self.parents = parents

        # attribute_name -> attribute_type
        self.attributedict = {}

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
