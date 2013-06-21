from contextlib import contextmanager
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
        raise Exception("At line %d: %s" % (lineno, e))
