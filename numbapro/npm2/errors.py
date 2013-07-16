from contextlib import contextmanager
import sys

class CompilerError(Exception):
    pass

@contextmanager
def error_context(lineno=-1, during=None):
    try:
        yield
    except Exception, e:
        msg = []
        if lineno >= 0:
            msg.append('At line %d:' % lineno)
        if during:
            msg.append('During: %s' % during)

        if isinstance(e, AssertionError):
            msg.append('Internal error: %s' % e)
        elif isinstance(e, CompilerError):
            msg.append(str(e))
        else:
            msg.append('%s: %s' % (type(e).__name__, e))

        exc = CompilerError('\n'.join(msg))
        raise exc, None, sys.exc_info()[2]
