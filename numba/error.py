# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import traceback

__all__ = ["NumbaError", "InternalError", "InvalidTemplateError"]

def format_pos(node):
    if node is not None and hasattr(node, 'lineno'):
        return format_postup((node.lineno, node.col_offset))
    else:
        return ""

def format_postup(tup):
    lineno, col_offset = tup
    return "%s:%s: " % (lineno, col_offset)

class NumbaError(Exception):
    "Some error happened during compilation"

    def __init__(self, node, msg=None, *args, **kwds):
        if msg is None:
            node, msg = None, node

        self.node = node
        self.msg = msg
        self.args = args
        self.has_report = kwds.get("has_report", False)

    def __str__(self):
        try:
            if self.has_report:
                return self.msg.strip()
            pos = format_pos(self.node)
            msg = "%s%s %s" % (pos, self.msg, " ".join(map(str, self.args)))
            return msg.rstrip()
        except:
            traceback.print_exc()
            return "<internal error creating numba error message>"


class InternalError(Exception):
    "Indicates a compiler bug"

class _UnknownAttribute(Exception):
    pass

class InvalidTemplateError(Exception):
    "Raised for invalid template type specifications"

class UnpromotableTypeError(TypeError):
    "Raised when we can't promote two given types"
    def __str__(self):
        return "Cannot promote types %s and %s" % self.args[0]
