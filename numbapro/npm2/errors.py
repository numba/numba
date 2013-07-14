from contextlib import contextmanager
import sys

@contextmanager
def error_context(lineno=-1, when=None):
    try:
        yield
    except Exception, e:
        msg = []
        if lineno >= 0:
            msg.append('At line %d:' % lineno)
        if isinstance(e, AssertionError):
            msg.append('Internal error: %s' % e)
        else:
            msg.append('%s: %s' % (type(e).__name__, e))
        if when:
            msg.append('when: %s' % when)

        exc = Exception('\n'.join(msg))
        raise exc, None, sys.exc_info()[2]
