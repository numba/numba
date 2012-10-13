"""
Define some errors that may be raised by the compiler.
"""

class Error(Exception):
    "Base exception class"

    def __repr__(self):
        return '%s()' % type(self).__name__

class InferTypeError(Error):
    "Raised when types of values cannot be inferred"

class UnmappableTypeError(Error):
    "Raised when a type cannot be mapped"

class UnpromotableTypeError(Error):
    "Raised when the compiler does not know how to promote two types."

class UnmappableFormatSpecifierError(Error):
    "Raised when a type cannot be mapped to a (printf) format specifier"

class InvalidTypeSpecification(Error):
    "Raised when a type is sliced incorrectly."

class CompileError(Error):
    "Raised for miscellaneous errors"

    def __init__(self, node, msg):
        self.node = node
        self.msg = msg

    def __str__(self):
        if self.node.pos is not None:
            return "%s:%s:%s: %s" % self.node.pos + self.msg
        return self.msg