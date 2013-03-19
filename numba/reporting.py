# -*- coding: utf-8 -*-
"""
Error reporting. Used by the CFA and by each FunctionEnvironment,
which can collect errors and warnings and issue them after failed or
successful compilation.
"""
from __future__ import print_function, division, absolute_import

import sys
import inspect

from numba import error

def getpos(node):
    try:
        return node.lineno, node.col_offset
    except:
        return 0, 0
# ______________________________________________________________________

class SourceDescr(object):
    """
    Source code descriptor.
    """

    def __init__(self, func, ast):
        self.func = func
        self.ast = ast

    def get_lines(self):
        source = None
        if self.func:
            try:
                source = inspect.getsource(self.func)
            except EnvironmentError:
                pass

        if source is None:
            try:
                from meta import asttools
                source = asttools.dump_python_source(self.ast)
            except Exception:
                source = ""

        first_lineno = getattr(self.ast, "lineno", 2)
        line_offset = offset(source.splitlines())

        newlines = "\n" * (first_lineno - line_offset)
        source = newlines + source

        return source.splitlines()

def offset(source_lines):
    offset = 0
    for line in source_lines:
        if line.strip().startswith("def"):
            break

        offset += 1

    return offset

# ______________________________________________________________________

def sort_message(collected_message):
    node, is_error, message = collected_message
    lineno, colno = float('inf'), float('inf')
    if hasattr(node, 'lineno'):
        lineno, colno = map(float, getpos(node))

    return not is_error, lineno, colno, message

class MessageCollection(object):
    """Collect error/warnings messages first then sort"""

    def __init__(self, ast=None, source_lines=None, file=None):
        # (node, is_error, message)
        self.file = file or sys.stdout
        self.ast = ast
        self.source_lines = source_lines
        self.messages = []

    def error(self, node, message):
        self.messages.append((node, True, message))

    def warning(self, node, message):
        self.messages.append((node, False, message))

    def header(self):
        pass

    def footer(self):
        pass

    def report_message(self, message, node, type):
        self.file.write(format_msg_simple(type, node, message))

    def report(self, post_mortem=False):
        self.messages.sort(key=sort_message)

        if self.messages:
            self.header()

        errors = []
        for node, is_error, message in self.messages:
            if is_error:
                errors.append((node, message))
                type = "Error"
            else:
                type = "Warning"

            self.report_message(message, node, type)

        if self.messages:
            self.footer()

        if errors and not post_mortem:
            raise error.NumbaError(*errors[0])

class FancyMessageCollection(MessageCollection):

    def header(self):
        self.file.write(" Numba Encountered Errors or Warnings ".center(80, "-") + '\n')

    def footer(self):
        self.file.write("-" * 80 + '\n')

    def report_message(self, message, node, type):
        self.file.write(format_msg(type, self.source_lines, node, message))

# ______________________________________________________________________

def format_msg(type, source_lines, node, msg):
    ret = ''
    if node and hasattr(node, 'lineno') and source_lines:
        lineno, colno = getpos(node)
        line = source_lines[lineno]

        ret = line + '\n' + "%s^" % ("-" * colno) + '\n'

    return ret + format_msg_simple(type, node, msg)

def format_msg_simple(type, node, message):
    "Issue a warning"
    # printing allows us to test the code
    return "%s %s%s\n" % (type, error.format_pos(node), message)
    # logger.warning("Warning %s: %s", error.format_postup(getpos(node)), message)

def warn_unreachable(node):
    "Generate a warning for unreachable code"
    if hasattr(node, 'lineno'):
        print("Warning, unreachable code at %s\n" % error.format_pos(node).rstrip(': '))

# ______________________________________________________________________

def report(function_error_env, post_mortem, exc=None):
    """
    :param function_error_env: the FunctionErrorEnvironment
    :param post_mortem: whether to enable post-mortem debugging of Numba
    :param exc: currently propagating exception
    """
    if exc is not None:
        function_error_env.collection.error(exc.node, exc.msg)

    try:
        function_error_env.collection.report(post_mortem)
    except error.NumbaError as e:
        if exc is None:
            exc = e

    if exc is not None and not post_mortem:
        # Shorten traceback
        raise exc
