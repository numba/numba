from contextlib import contextmanager
import sys

class CompileError(Exception):
    def __init__(self, lineno, msg):
        if lineno != -1:
            msg = "At line %d: %s" % (lineno, msg)
        super(CompileError, self).__init__(msg)

@contextmanager
def error_context(lineno):
    try:
        yield
    except Exception, e:
        if lineno < 0:
            lineno = "?"
        exc = Exception("Caused by input line %s: %s\n%s" % (lineno, type(e), e))
        raise exc, None, sys.exc_info()[2]
