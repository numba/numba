# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import warnings
from functools import partial

from numba.config import config

try:
    import pygments
except ImportError, e:
    warnings.warn("Pygments not installed")
    pygments = None
else:
    from pygments import highlight
    from pygments.lexers import PythonLexer, LlvmLexer
    from pygments.formatters import HtmlFormatter, TerminalFormatter

# ______________________________________________________________________

lexers = {
    "python": PythonLexer,
    "llvm": LlvmLexer,
}

formatters = {
    "html": HtmlFormatter,
    "console": partial(TerminalFormatter, bg=config.terminal_background),
}

def lex_source(code, lexer="python", output='html', inline_css=True):
    if not config.colour:
        return code

    Lexer = lexers[lexer]
    Formatter = formatters[output]
    result = highlight(code, Lexer(), Formatter(noclasses=inline_css))
    return result.rstrip()