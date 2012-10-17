import traceback

class NumbaError(Exception):
    "Some error happened during compilation"

    def __init__(self, node, msg=None, *args):
        if msg is None:
            node, msg = None, node

        self.node = node
        self.msg = msg
        self.args = args

    def __str__(self):
        try:
            pos = ""
            if self.node is not None and hasattr(self.node, 'lineno'):
                pos = "%s:%s: " % (self.node.lineno, self.node.col_offset)

            return "%s%s %s" % (pos, self.msg, " ".join(map(str, self.args)))
        except:
            traceback.print_exc()
            return ""


class InternalError(NumbaError):
    "Indicates a compiler bug"

class _UnknownAttribute(Exception):
    pass