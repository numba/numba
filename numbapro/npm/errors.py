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
        raise Exception("At line %s: %s" % (lineno, e)), None, sys.exc_info()[2]
