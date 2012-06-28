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

    def __init__(self, type, is_local=False, is_global=False,
                 is_constant=False, name=None, lvalue=None):
        self.name = name
        self.type = type
        self.is_local = is_local
        self.is_constant = is_constant
        self.is_global = is_global
        self.lvalue = lvalue

    @classmethod
    def from_variable(cls, variable, **kwds):
        result = cls(variable.type)
        vars(result).update(kwds)
        return result

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