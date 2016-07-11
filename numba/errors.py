"""
Numba-specific errors and warnings.
"""

from __future__ import print_function, division, absolute_import

import sys
import contextlib
from collections import defaultdict
import warnings
from numba import six
from functools import wraps


# Filled at the end
__all__ = []

def deprecated(arg):
    """Define a deprecation decorator.
    An optional string should refer to the new API to be used instead.

    Example:
      @deprecated
      def old_func(): ...

      @deprecated('new_func')
      def old_func(): ..."""

    subst = arg if isinstance(arg, str) else None
    def decorator(func):
        def wrapper(*args, **kwargs):
            msg = "Call to deprecated function \"{}\"."
            if subst:
                msg += "\n Use \"{}\" instead."
            warnings.warn(msg.format(func.__name__, subst),
                          category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wraps(func)(wrapper)

    if not subst:
        return decorator(arg)
    else:
        return decorator


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
    @property
    def contexts(self):
        try:
            return self._contexts
        except AttributeError:
            self._contexts = lst = []
            return lst

    def add_context(self, msg):
        """
        Add contextual info.  The execption message is expanded with the new
        contextual info.
        """
        self.contexts.append(msg)
        newmsg = '{0}\n[{1}] During: {2}'.format(self, len(self.contexts), msg)
        self.args = (newmsg,)
        return self

    def patch_message(self, new_message):
        """
        Change the error message to the given new message.
        """
        self.args = (new_message,) + self.args[1:]


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
        msg = "Unknown attribute '{attr}' of type {type}".format(type=value,
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


class InternalError(NumbaError):
    """
    For wrapping internal error occured within the compiler
    """
    def __init__(self, exception):
        super(InternalError, self).__init__(str(exception))
        self.old_exception = exception


def _format_msg(fmt, args, kwargs):
    return fmt.format(*args, **kwargs)


@contextlib.contextmanager
def new_error_context(fmt_, *args, **kwargs):
    """
    A contextmanager that prepend contextual information to any exception
    raised within.  If the exception type is not an instance of NumbaError,
    it will be wrapped into a InternalError.   The exception class can be
    changed by providing a "errcls_" keyword argument with the exception
    constructor.

    The first argument is a message that describes the context.  It can be a
    format string.  If there are additional arguments, it will be used as
    ``fmt_.format(*args, **kwargs)`` to produce the final message string.
    """
    errcls = kwargs.pop('errcls_', InternalError)
    try:
        yield
    except NumbaError as e:
        e.add_context(_format_msg(fmt_, args, kwargs))
        raise
    except Exception as e:
        newerr = errcls(e).add_context(_format_msg(fmt_, args, kwargs))
        six.reraise(type(newerr), newerr, sys.exc_info()[2])


__all__ += [name for (name, value) in globals().items()
            if not name.startswith('_') and isinstance(value, type)
               and issubclass(value, (Exception, Warning))]
