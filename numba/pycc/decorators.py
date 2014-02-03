from __future__ import print_function, absolute_import
import re
import inspect
from numba import sigutils

# Registry is okay to be a global because we are using pycc as a standalone
# commandline tool.
registry = []


def export(prototype):
    sym, sig = parse_prototype(prototype)

    def wrappped(func):
        module = inspect.getmodule(func).__name__
        signature = sigutils.parse_signature(sig)
        entry = ExportEntry(symbol=sym, signature=signature, function=func,
                            module=module)
        registry.append(entry)

    return wrappped


def exportmany(prototypes):
    def wrapped(func):
        for proto in prototypes:
            export(proto)(func)
    return wrapped

# --------------------------------- Internal ---------------------------------

re_symbol = re.compile(r'[_a-z][_a-z0-9]*', re.I)


class ExportEntry(object):
    """
    A simple record for exporting symbols.
    """

    def __init__(self, symbol, signature, function, module):
        self.symbol = symbol
        self.signature = signature
        self.function = function
        self.module = module

    def __repr__(self):
        return "ExportEntry('%s', '%s')" % (self.symbol, self.signature)


def parse_prototype(text):
    """Separate the symbol and function-type in a a string with
    "symbol function-type" (e.g. "mult float(float, float)")

    Returns
    ---------
    (symbol_string, functype_string)
    """
    m = re_symbol.match(text)
    if not m:
        raise ValueError("Invalid function name for export prototype")
    s = m.start(0)
    e = m.end(0)
    symbol = text[s:e]
    functype = text[e + 1:]
    return symbol, functype

