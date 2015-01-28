"""
Itanium CXX ABI Mangler

Reference: http://mentorembedded.github.io/cxx-abi/abi.html
"""

from __future__ import print_function, absolute_import
from numba import types
import re


PREFIX = "_Z"

C2CODE = {
    'void': 'v',
    'wchar_t': 'w',
    'bool': 'b',
    'char': 'c',
    'signed char': 'a',
    'unsigned char': 'h',
    'short': 's',
    'unsigned short': 't',
    'int': 'i',
    'unsigned int': 'j',
    'long': 'l',
    'unsigned long': 'm',
    'long long': 'x', # __int64
    'unsigned long long': 'y', # unsigned __int64
    '__int128': 'n',
    'unsigned __int128': 'o',
    'float': 'f',
    'double': 'd',
    'long double': 'e', # __float80
    '__float128': 'g',
    'ellipsis': 'z',
}

N2C = {
    types.void: 'void',
    types.boolean: 'bool',
    types.uint8: 'unsigned char',
    types.int8: 'signed char',
    types.uint16: 'unsigned short',
    types.int16: 'short',
    types.uint32: 'unsigned int',
    types.int32: 'int',
    types.uint64: 'unsigned long long',
    types.int64: 'long long',
    types.float32: 'float',
    types.float64: 'double',
}


def _encode(ch):
    """
    Encode a single character.
    Anything not valid as Python identifier is encoded as '%N' where N
    is a decimal number of the character code point.
    """
    if ch.isalnum() or ch in '_':
        out = ch
    else:
        out = "%%%d" % ord(ch)
    return out


def mangle_identifier(ident):
    """
    Mangle the identifier

    This treats '.' as '::' in C++
    """
    splitted = (''.join(map(_encode, x)) for x in ident.split('.'))
    parts = ["%d%s" % (len(x), x) for x in splitted]
    if len(parts) > 1:
        return 'N%sE' % ''.join(parts)
    else:
        return parts[0]


def mangle_type_c(typ):
    """
    Mangle C type name

    Args
    ----
    typ: str
        C type name
    """
    if typ in C2CODE:
        return C2CODE[typ]
    else:
        return 'u' + mangle_identifier(typ)


def mangle_type(typ):
    """
    Mangle Numba type
    """
    if typ in N2C:
        typename = N2C[typ]
    else:
        typename = str(typ)
    return mangle_type_c(typename)


def mangle_args_c(argtys):
    """
    Mangle sequence of C type names
    """
    return ''.join([mangle_type_c(t) for t in argtys])


def mangle_args(argtys):
    """
    Mangle sequence of Numba type objects
    """
    return ''.join([mangle_type(t) for t in argtys])


def mangle_c(ident, argtys):
    """
    Mangle identifier with C type names
    """
    return PREFIX + mangle_identifier(ident) + mangle_args_c(argtys)


def mangle(ident, argtys):
    """
    Mangle identifier with Numba type objects
    """
    return PREFIX + mangle_identifier(ident) + mangle_args(argtys)
