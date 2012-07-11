from . import utils

class Variable(object):
    """
    Variables placed on the stack. They allow an indirection
    so, that when used in an operation, the correct LLVM type can be inserted.

    Attributes:
        type: the Numba type (see _numba_types and minivect/minitypes)
        is_local/is_global/is_constant
        name: name of local or global
        lvalue: LLVM Value
        state: state passed from one stage to the next
    """

    def __init__(self, type, is_constant=False, is_local=False,
                 name=None, lvalue=None, constant_value=None):
        self.name = name
        self.type = type
        self.is_constant = is_constant
        self.constant_value = constant_value
        self.lvalue = lvalue

    @classmethod
    def from_variable(cls, variable, **kwds):
        result = cls(variable.type)
        vars(result).update(kwds)
        return result

    @property
    def is_local(self):
        return not self.is_global

    @property
    def is_global(self):
        return self.type and self.type.is_global

    @property
    def ltype(self):
        """
        The LLVM type for the type of this variable or LLVM Value.
        """
        if self.lvalue is not None:
            return self.lvalue.type
        return self.type.to_llvm(utils.context)

    @property
    def ctypes_type(self):
        """
        The ctypes type for the type of this variable.
        """

    def __repr__(self):
        args = []
        if self.is_local:
            args.append("is_local=True")
        if self.is_global:
            args.append("is_global=True")
        if self.is_constant:
            args.append("is_constant=True")
        if self.name:
            args.append("name=%s" % self.name)
        if self.lvalue:
            args.append("llvm=%s" % (self.lvalue,))

        if args:
            extra_info = " " + ", ".join(args)
        else:
            extra_info = ""

        return '<Variable(type=%s%s)>' % (self.type, extra_info)