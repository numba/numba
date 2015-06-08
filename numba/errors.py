
__all__ = []


class NumbaError(Exception):
    pass


class IRError(NumbaError):
    """
    An error occurred during Numba IR generation.
    """

class RedefinedError(IRError):
    pass

class NotDefinedError(IRError):
    def __init__(self, name, loc=None):
        self.name = name
        self.loc = loc

    def __str__(self):
        loc = "?" if self.loc is None else self.loc
        return "{name!r} is not defined in {loc}".format(name=self.name,
                                                         loc=self.loc)

class VerificationError(IRError):
    pass


class MacroError(NumbaError):
    """
    An error occurred during macro expansion.
    """


class DeprecationError(NumbaError):
    pass


class LoweringError(NumbaError):
    """
    An error occurred during lowering.
    """

    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


class ForbiddenConstruct(LoweringError):
    """
    A forbidden Python construct was encountered (e.g. use of locals()).
    """


class TypingError(NumbaError):
    """
    A type inference failure.
    """
    def __init__(self, msg, loc=None):
        self.msg = msg
        self.loc = loc
        if loc:
            super(TypingError, self).__init__("%s\n%s" % (msg, loc.strformat()))
        else:
            super(TypingError, self).__init__("%s" % (msg,))


class UntypedAttributeError(TypingError):
    def __init__(self, value, attr):
        msg = 'Unknown attribute "{attr}" of type {type}'.format(type=value,
                                                              attr=attr)
        super(UntypedAttributeError, self).__init__(msg)


class ByteCodeSupportError(NumbaError):
    """
    Failure to extract the bytecode of the user's function.
    """


class CompilerError(NumbaError):
    """
    Some high-level error in the compiler.
    """


__all__ += [name for (name, value) in globals().items()
            if not name.startswith('_') and issubclass(value, Exception)]
