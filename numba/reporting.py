"""
Error reporting. Used by the CFA and by each FunctionEnvironment,
which can collect errors and warnings and issue them after failed or
successful compilation.
"""

import inspect

from numba import error

def getpos(node):
    return node.lineno, node.col_offset

class SourceDescr(object):
    """
    Source code descriptor.
    """

    def __init__(self, func, ast):
        self.func = func
        self.ast = ast

    def get_lines(self):
        if self.func:
            source = inspect.getsource(self.func)
        else:
            try:
                from meta import asttools
                source = asttools.dump_python_source(self.ast)
            except Exception:
                source = ""

        source = "\n" * (self.ast.lineno - 2) + source
        return source.splitlines()

class MessageCollection(object):
    """Collect error/warnings messages first then sort"""

    def __init__(self):
        # (node, is_error, message)
        self.messages = []

    def error(self, node, message):
        self.messages.append((node, True, message))

    def warning(self, node, message):
        self.messages.append((node, False, message))

    def report(self, post_mortem=False):
        self.messages.sort()
        errors = []
        for node, is_error, message in self.messages:
            if is_error:
                errors.append((node, message))
            warning(node, message)

        if errors and not post_mortem:
            raise error.NumbaError(*errors[0])

def warning(node, message):
    "Issue a warning"
    # printing allows us to test the code
    print "Warning %s%s" % (error.format_pos(node), message)
    # logger.warning("Warning %s: %s", error.format_postup(getpos(node)), message)

def warn_unreachable(node):
    "Generate a warning for unreachable code"
    if hasattr(node, 'lineno'):
        print "Warning, unreachable code at %s" % error.format_pos(node).rstrip(': ')


def report(function_error_env, post_mortem, exc=None):
    """
    :param function_error_env: the FunctionErrorEnvironment
    :param post_mortem: whether to enable post-mortem debugging of Numba
    :param exc: currently propagating exception
    """
    if exc is not None:
        function_error_env.collection.error(exc.node, exc.msg)

    try:
        function_error_env.collection.report(post_mortem)
    except NumbaError as e:
        if exc is None:
            exc = e

    if exc is not None and not post_mortem:
        # Shorten traceback
        raise exc