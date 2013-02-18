import inspect

from numba import error

def getpos(node):
    return node.lineno, node.col_offset

class SourceDescr(object):
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

class MessageCollection:
    """Collect error/warnings messages first then sort"""

    def __init__(self):
        self.messages = []

    def error(self, node, message):
        self.messages.append((getpos(node), node, True, message))

    def warning(self, node, message):
        self.messages.append((getpos(node), node, False, message))

    def report(self):
        self.messages.sort()
        errors = []
        for pos, node, is_error, message in self.messages:
            if is_error:
                errors.append((node, message))
            warning(node, message)

        if errors:
            raise error.NumbaError(*errors[0])

def warning(node, message):
    # printing allows us to test the code
    print "Warning %s%s" % (error.format_pos(node), message)
    # logger.warning("Warning %s: %s", error.format_postup(getpos(node)), message)

def warn_unreachable(node):
    if hasattr(node, 'lineno'):
        print "Warning, unreachable code at %s" % error.format_pos(node).rstrip(': ')
