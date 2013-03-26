"""
Extension attribute Variables used for attribute type inference.
See also compileclass.build_extension_symtab().
"""

from numba import symtab

class ExtensionAttributeVariable(symtab.Variable):
    """
    Variable created during type inference for assignments to extension
    attributes for which we don't know the type yet.

    When the assignment happens, update ext_type.attributedict.
    """

    def __init__(self, ext_type, attr_name, type, *args, **kwargs):
        super(ExtensionAttributeVariable, self).__init__(type, *args, **kwargs)
        self.ext_type = ext_type
        self.attr_name = attr_name

    def perform_assignment(self, rhs_type):
        self.type = rhs_type
        self.ext_type.attributedict[self.attr_name] = rhs_type
