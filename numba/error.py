import traceback

from numba.minivect.minierror import Error

__all__ = ["Error", "NumbaError", "InternalError", "InvalidTemplateError"]

def format_pos(node):
    if node is not None and hasattr(node, 'lineno'):
        return format_postup((node.lineno, node.col_offset))
    else:
        return ""

def format_postup(tup):
    lineno, col_offset = tup
    return "%s:%s: " % (lineno - 1, col_offset)

class NumbaError(Error):
    "Some error happened during compilation"

    def __init__(self, node, msg=None, *args):
        if msg is None:
            node, msg = None, node

        self.node = node
        self.msg = msg
        self.args = args

    def __str__(self):
        try:
            pos = format_pos(self.node)
            msg = "%s%s %s" % (pos, self.msg, " ".join(map(str, self.args)))
            return msg.rstrip()
        except:
            traceback.print_exc()
            return "<internal error creating numba error message>"


class InternalError(Error):
    "Indicates a compiler bug"

class _UnknownAttribute(Error):
    pass

class InvalidTemplateError(Error):
    "Raised for invalid template type specifications"