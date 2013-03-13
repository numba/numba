

from numba.typesystem import *

#------------------------------------------------------------------------
# Extension Method Types
#------------------------------------------------------------------------

class ExtMethodType(NumbaType, minitypes.FunctionType):
    """
    Extension method type, a FunctionType plus the following fields:

        is_class_method: is classmethod?
        is_static_method: is staticmethod?
        is_bound_method: is bound method?
    """

    def __init__(self, return_type, args, name=None,
                 is_class=False, is_static=False, **kwds):
        super(ExtMethodType, self).__init__(return_type, args, name, **kwds)

        self.is_class_method = is_class
        self.is_static_method = is_static
        self.is_bound_method = not (is_class or is_static)
