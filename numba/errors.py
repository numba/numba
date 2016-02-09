"""
Numba-specific errors and warnings.
"""

from __future__ import print_function, division, absolute_import

import contextlib
from collections import defaultdict
import warnings


# Filled at the end
__all__ = []


class NumbaWarning(Warning):
    """
    Base category for all Numba compiler warnings.
    """

class PerformanceWarning(NumbaWarning):
    """
    Warning category for when an operation might not be
    as fast as expected.
    """


class WarningsFixer(object):
    """
    An object "fixing" warnings of a given category caught during
    certain phases.  The warnings can have their filename and lineno fixed,
    and they are deduplicated as well.
    """

    def __init__(self, category):
        self._category = category
        # {(filename, lineno, category) -> messages}
        self._warnings = defaultdict(set)

    @contextlib.contextmanager
    def catch_warnings(self, filename=None, lineno=None):
        """
        Store warnings and optionally fix their filename and lineno.
        """
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always', self._category)
            yield

        for w in wlist:
            msg = str(w.message)
            if issubclass(w.category, self._category):
                # Store warnings of this category for deduplication
                filename = filename or w.filename
                lineno = lineno or w.lineno
                self._warnings[filename, lineno, w.category].add(msg)
            else:
                # Simply emit other warnings again
                warnings.warn_explicit(msg, w.category,
                                       w.filename, w.lineno)

    def flush(self):
        """
        Emit all stored warnings.
        """
        for (filename, lineno, category), messages in sorted(self._warnings.items()):
            for msg in sorted(messages):
                warnings.warn_explicit(msg, category, filename, lineno)
        self._warnings.clear()


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
    def __init__(self, value, attr, loc=None):
        msg = 'Unknown attribute "{attr}" of type {type}'.format(type=value,
                                                                 attr=attr)
        super(UntypedAttributeError, self).__init__(msg, loc=loc)


class ByteCodeSupportError(NumbaError):
    """
    Failure to extract the bytecode of the user's function.
    """


class CompilerError(NumbaError):
    """
    Some high-level error in the compiler.
    """


class ConstantInferenceError(NumbaError):
    """
    Failure during constant inference.
    """


__all__ += [name for (name, value) in globals().items()
            if not name.startswith('_') and isinstance(value, type)
               and issubclass(value, (Exception, Warning))]
