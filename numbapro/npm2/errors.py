from contextlib import contextmanager
import sys

@contextmanager
def error_context(lineno=-1):
    try:
        yield
    except Exception, e:
        if lineno < 0:
            lineno = "?"
        msg = "Caused by input line %s: %s\n%s" % (lineno, type(e), e)
        if isinstance(e, AssertionError):
            msg = "Internal error: %s" % msg
        exc = Exception(msg)
        raise exc, None, sys.exc_info()[2]
