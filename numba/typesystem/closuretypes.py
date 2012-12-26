"""
Types for closures and inner functions.
"""

from numba.typesystem import *

class ClosureType(NumbaType, minitypes.ObjectType):
    """
    Type of closures and inner functions.
    """

    is_closure = True

    def __init__(self, signature, **kwds):
        super(ClosureType, self).__init__(**kwds)
        self.signature = signature
        self.closure = None

    def __repr__(self):
        return "<closure(%s)>" % self.signature

class ClosureScopeType(ExtensionType):
    """
    Type of the enclosing scope for closures. This is always passed in as
    first argument to the function.
    """

    is_closure_scope = True
    is_final = True

    def __init__(self, py_class, parent_scope, **kwds):
        super(ClosureScopeType, self).__init__(py_class, **kwds)
        self.parent_scope = parent_scope
        self.unmangled_symtab = None

        if self.parent_scope is None:
            self.scope_prefix = ""
        else:
            self.scope_prefix = self.parent_scope.scope_prefix + "0"

    def __repr__(self):
        return "closure_scope(%s)" % self.attribute_struct