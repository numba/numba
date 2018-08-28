"""
Numba-specific errors and warnings.
"""

from __future__ import print_function, division, absolute_import

import abc
import contextlib
import os
import sys
import warnings
import numba
import numpy as np
from collections import defaultdict
from numba import six
from functools import wraps
from abc import abstractmethod

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


class ParallelSafetyWarning(RuntimeWarning):
    """
    Warning category for when an operation in a prange
    might not have parallel semantics.
    """

# These are needed in the color formatting of errors setup

@six.add_metaclass(abc.ABCMeta)
class _ColorScheme(object):

    @abstractmethod
    def code(self, msg):
        pass

    @abstractmethod
    def errmsg(self, msg):
        pass

    @abstractmethod
    def filename(self, msg):
        pass

    @abstractmethod
    def indicate(self, msg):
        pass

    @abstractmethod
    def highlight(self, msg):
        pass


class _DummyColorScheme(_ColorScheme):

    def __init__(self, theme=None):
        pass

    def code(self, msg):
        pass

    def errmsg(self, msg):
        pass

    def filename(self, msg):
        pass

    def indicate(self, msg):
        pass

    def highlight(self, msg):
        pass


# holds reference to the instance of the terminal color scheme in use
_termcolor_inst = None

try:
    import colorama

    # If the colorama version is < 0.3.9 it can break stdout/stderr in some
    # situations, as a result if this condition is met colorama is disabled and
    # the user is warned.
    if tuple([int(x) for x in colorama.__version__.split('.')]) < (0, 3, 9):
        msg = ("Insufficiently recent colorama version found. "
               "Numba requires colorama >= 0.3.9")
        # warn the user
        warnings.warn(msg)
        # trip the exception to disable color errors
        raise ImportError

    # If Numba is running in testsuite mode then do not use error message
    # coloring so CI system output is consistently readable without having
    # to read between shell escape characters.
    if os.environ.get('NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING', None):
        raise ImportError # just to trigger the exception handler below

except ImportError:

    class NOPColorScheme(_DummyColorScheme):
        def __init__(self, theme=None):
            if theme is not None:
                raise ValueError("specifying a theme has no effect")
            _DummyColorScheme.__init__(self, theme=theme)

        def code(self, msg):
            return msg

        def errmsg(self, msg):
            return msg

        def filename(self, msg):
            return msg

        def indicate(self, msg):
            return msg

        def highlight(self, msg):
            return msg

    def termcolor():
        global _termcolor_inst
        if _termcolor_inst is None:
            _termcolor_inst = NOPColorScheme()
        return _termcolor_inst

else:

    from colorama import init, reinit, deinit, Fore, Back, Style
    from contextlib import contextmanager

    class ColorShell(object):
        _has_initialized = False

        def __init__(self):
            init()
            self._has_initialized = True

        def __enter__(self):
            if self._has_initialized:
                reinit()

        def __exit__(self, *exc_detail):
            Style.RESET_ALL
            deinit()

    class reset_terminal(object):
        def __init__(self):
            self._buf = bytearray(b'')

        def __enter__(self):
            return self._buf

        def __exit__(self, *exc_detail):
            self._buf += bytearray(Style.RESET_ALL.encode('utf-8'))

    # define some default themes, if more are added, update the envvars docs!
    themes = {}

    # No color added, just bold weighting
    themes['no_color'] = {'code': None,
                         'errmsg': None,
                         'filename': None,
                         'indicate': None,
                         'highlight': None, }

    # suitable for terminals with a dark background
    themes['dark_bg'] = {'code': Fore.BLUE,
                         'errmsg': Fore.YELLOW,
                         'filename': Fore.WHITE,
                         'indicate': Fore.GREEN,
                         'highlight': Fore.RED, }

    # suitable for terminals with a light background
    themes['light_bg'] = {'code': Fore.BLUE,
                          'errmsg': Fore.BLACK,
                          'filename': Fore.MAGENTA,
                          'indicate': Fore.BLACK,
                          'highlight': Fore.RED, }

    # suitable for terminals with a blue background
    themes['blue_bg'] = {'code': Fore.WHITE,
                         'errmsg': Fore.YELLOW,
                         'filename': Fore.MAGENTA,
                         'indicate': Fore.CYAN,
                         'highlight': Fore.RED, }

    # suitable for use in jupyter notebooks
    themes['jupyter_nb'] = {'code': Fore.BLACK,
                         'errmsg': Fore.BLACK,
                         'filename': Fore.GREEN,
                         'indicate': Fore.CYAN,
                         'highlight': Fore.RED, }

    default_theme = themes['no_color']

    class HighlightColorScheme(_DummyColorScheme):
        def __init__(self, theme=default_theme):
            self._code = theme['code']
            self._errmsg = theme['errmsg']
            self._filename = theme['filename']
            self._indicate = theme['indicate']
            self._highlight = theme['highlight']
            _DummyColorScheme.__init__(self, theme=theme)

        def _markup(self, msg, color=None, style=Style.BRIGHT):
            features = ''
            if color:
                features += color
            if style:
                features += style
            with ColorShell():
                with reset_terminal() as mu:
                    mu += features.encode('utf-8')
                    mu += (msg).encode('utf-8')
                return mu.decode('utf-8')

        def code(self, msg):
            return self._markup(msg, self._code)

        def errmsg(self, msg):
            return self._markup(msg, self._errmsg)

        def filename(self, msg):
            return self._markup(msg, self._filename)

        def indicate(self, msg):
            return self._markup(msg, self._indicate)

        def highlight(self, msg):
            return self._markup(msg, self._highlight)

    def termcolor():
        global _termcolor_inst
        if _termcolor_inst is None:
            scheme = themes[numba.config.COLOR_SCHEME]
            _termcolor_inst = HighlightColorScheme(scheme)
        return _termcolor_inst


unsupported_error_info = """
Unsupported functionality was found in the code Numba was trying to compile.

If this functionality is important to you please file a feature request at:
https://github.com/numba/numba/issues/new
"""

interpreter_error_info = """
Unsupported Python functionality was found in the code Numba was trying to
compile. This error could be due to invalid code, does the code work
without Numba? (To temporarily disable Numba JIT, set the `NUMBA_DISABLE_JIT`
environment variable to non-zero, and then rerun the code).

If the code is valid and the unsupported functionality is important to you
please file a feature request at: https://github.com/numba/numba/issues/new

To see Python/NumPy features supported by the latest release of Numba visit:
http://numba.pydata.org/numba-doc/dev/reference/pysupported.html
and
http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
"""

constant_inference_info = """
Numba could not make a constant out of something that it decided should be
a constant. This could well be a current limitation in Numba's internals,
please either raise a bug report along with a minimal reproducer at:
https://github.com/numba/numba/issues/new
"""

typing_error_info = """
This is not usually a problem with Numba itself but instead often caused by
the use of unsupported features or an issue in resolving types.

To see Python/NumPy features supported by the latest release of Numba visit:
http://numba.pydata.org/numba-doc/dev/reference/pysupported.html
and
http://numba.pydata.org/numba-doc/dev/reference/numpysupported.html

For more information about typing errors and how to debug them visit:
http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-doesn-t-compile

If you think your code should work with Numba, please report the error message
and traceback, along with a minimal reproducer at:
https://github.com/numba/numba/issues/new
"""

reportable_issue_info = """
-------------------------------------------------------------------------------
This should not have happened, a problem has occurred in Numba's internals.

Please report the error message and traceback, along with a minimal reproducer
at: https://github.com/numba/numba/issues/new

If more help is needed please feel free to speak to the Numba core developers
directly at: https://gitter.im/numba/numba

Thanks in advance for your help in improving Numba!
"""

error_extras = dict()
error_extras['unsupported_error'] = unsupported_error_info
error_extras['typing'] = typing_error_info
error_extras['reportable'] = reportable_issue_info
error_extras['interpreter'] = interpreter_error_info
error_extras['constant_inference'] = constant_inference_info


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

    def __init__(self, msg, loc=None, highlighting=True):
        self.msg = msg
        self.loc = loc
        if highlighting:
            highlight = termcolor().errmsg
        else:
            def highlight(x): return x
        if loc:
            super(NumbaError, self).__init__(
                highlight("%s\n%s\n" % (msg, loc.strformat())))
        else:
            super(NumbaError, self).__init__(highlight("%s" % (msg,)))

    @property
    def contexts(self):
        try:
            return self._contexts
        except AttributeError:
            self._contexts = lst = []
            return lst

    def add_context(self, msg):
        """
        Add contextual info.  The exception message is expanded with the new
        contextual information.
        """
        self.contexts.append(msg)
        f = termcolor().errmsg('{0}\n') + termcolor().filename(
            '[{1}] During: {2}')
        newmsg = f.format(self, len(self.contexts), msg)
        self.args = (newmsg,)
        return self

    def patch_message(self, new_message):
        """
        Change the error message to the given new message.
        """
        self.args = (new_message,) + self.args[1:]


class UnsupportedError(NumbaError):
    """
    Numba does not have an implementation for this functionality.
    """
    pass


class IRError(NumbaError):
    """
    An error occurred during Numba IR generation.
    """
    pass


class RedefinedError(IRError):
    """
    An error occurred during interpretation of IR due to variable redefinition.
    """
    pass


class NotDefinedError(IRError):
    """
    An undefined variable is encountered during interpretation of IR.
    """
    def __init__(self, name, loc=None):
        self.name = name
        msg = "Variable '%s' is not defined." % name
        super(NotDefinedError, self).__init__(msg, loc=loc)


class VerificationError(IRError):
    """
    An error occurred during IR verification. Once Numba's internal
    representation (IR) is constructed it is then verified to ensure that
    terminators are both present and in the correct places within the IR. If
    it is the case that this condition is not met, a VerificationError is
    raised.
    """
    pass


class MacroError(NumbaError):
    """
    An error occurred during macro expansion.
    """
    pass


class DeprecationError(NumbaError):
    """
    Functionality is deprecated.
    """
    pass


class LoweringError(NumbaError):
    """
    An error occurred during lowering.
    """
    def __init__(self, msg, loc=None):
        super(LoweringError, self).__init__(msg, loc=loc)


class ForbiddenConstruct(LoweringError):
    """
    A forbidden Python construct was encountered (e.g. use of locals()).
    """
    pass


class TypingError(NumbaError):
    """
    A type inference failure.
    """
    pass


class UntypedAttributeError(TypingError):
    def __init__(self, value, attr, loc=None):
        module = getattr(value, 'pymod', None)
        if module is not None and module == np:
            # unsupported numpy feature.
            msg = ("Use of unsupported NumPy function 'numpy.%s' "
                   "or unsupported use of the function.") % attr
        else:
            msg = "Unknown attribute '{attr}' of type {type}"
            msg = msg.format(type=value, attr=attr)
        super(UntypedAttributeError, self).__init__(msg, loc=loc)


class ByteCodeSupportError(NumbaError):
    """
    Failure to extract the bytecode of the user's function.
    """
    def __init__(self, msg, loc=None):
        super(ByteCodeSupportError, self).__init__(msg, loc=loc)


class CompilerError(NumbaError):
    """
    Some high-level error in the compiler.
    """
    pass


class ConstantInferenceError(NumbaError):
    """
    Failure during constant inference.
    """
    def __init__(self, value, loc=None):
        self.value = value
        msg = "Cannot make a constant from: %s" % value
        super(ConstantInferenceError, self).__init__(msg, loc=loc)


class InternalError(NumbaError):
    """
    For wrapping internal error occured within the compiler
    """

    def __init__(self, exception):
        super(InternalError, self).__init__(str(exception))
        self.old_exception = exception


class RequireConstValue(TypingError):
    """For signaling a function typing require constant value for some of
    its arguments.
    """
    pass


def _format_msg(fmt, args, kwargs):
    return fmt.format(*args, **kwargs)


import os.path
_numba_path = os.path.dirname(__file__)
loc_info = {}


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

    loc = kwargs.get('loc', None)
    if loc is not None and not loc.filename.startswith(_numba_path):
        loc_info.update(kwargs)

    try:
        yield
    except NumbaError as e:
        e.add_context(_format_msg(fmt_, args, kwargs))
        raise
    except Exception as e:
        newerr = errcls(e).add_context(_format_msg(fmt_, args, kwargs))
        from numba import config
        tb = sys.exc_info()[2] if config.FULL_TRACEBACKS else None
        six.reraise(type(newerr), newerr, tb)


__all__ += [name for (name, value) in globals().items()
            if not name.startswith('_') and isinstance(value, type)
            and issubclass(value, (Exception, Warning))]
