

from numba.typesystem import *

#------------------------------------------------------------------------
# Extension Method Types
#------------------------------------------------------------------------

class ExtMethodType(NumbaType, minitypes.FunctionType):
    """
    Extension method type, a FunctionType plus the following fields:

        is_class: is classmethod?
        is_static: is staticmethod?
    """

    def __init__(self, return_type, args, name=None,
                 is_class=False, is_static=False, **kwds):
        super(ExtMethodType, self).__init__(return_type, args, name, **kwds)
        self.is_class = is_class
        self.is_static = is_static
