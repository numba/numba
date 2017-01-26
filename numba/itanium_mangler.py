"""
Itanium CXX ABI Mangler

Reference: http://mentorembedded.github.io/cxx-abi/abi.html
"""

from __future__ import print_function, absolute_import

import re

from numba import types


_re_invalid_char = re.compile(r'[^a-z0-9_]', re.I)


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


def _escape_string(text):
    """Escape the given string so that it only contains ASCII characters
    of [a-zA-Z0-9_$].

    The dollar symbol ($) and other invalid characters are escaped into
    the string sequence of "$xx" where "xx" is the hex codepoint of the char.

    Multibyte characters are encoded into utf8 and converted into the above
    hex format.
    """
    def repl(m):
        return ''.join(('$%02x' % ch) for ch in m.group(0).encode('utf8'))
    return re.sub(_re_invalid_char, repl, text)


def _fix_lead_digit(text):
    """
    Fix text with leading digit
    """
    if text and text[0].isdigit():
        return '_' + text[0]
    else:
        return text


def mangle_identifier(ident, template_params=''):
    """
    Mangle the identifier

    This treats '.' as '::' in C++
    """

    splitted = [_escape_string(x) for x in ident.split('.')]
    parts = ["%d%s" % (len(x), x) for x in map(_fix_lead_digit, splitted)]
    if len(parts) > 1:
        return 'N%s%sE' % (''.join(parts), template_params)
    else:
        return '%s%s' % (parts[0], template_params)


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
        return mangle_identifier(typ)


def mangle_type(typ):
    """
    Mangle type parameter.
    """
    # Handle numba types
    if isinstance(typ, types.Type):
        if typ in N2C:
            return mangle_type_c(N2C[typ])
        else:
            return mangle_templated_ident(*typ.mangling_args)
    # Handle integer literal
    elif isinstance(typ, int):
        return 'Li%dE' % typ
    # Otherwise
    return mangle_identifier(str(typ))


def mangle_templated_ident(identifier, parameters):
    if parameters:
        template_params = 'I%sE' % ''.join(map(mangle_type, parameters))
    else:
        template_params = ''
    return mangle_identifier(identifier, template_params)


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


def prepend_namespace(mangled, ns):
    """
    Prepend namespace to mangled name.
    """
    assert mangled.startswith(PREFIX)
    if mangled.startswith(PREFIX + 'N'):
        # nested
        remaining = mangled[3:]
        ret = PREFIX + 'N' + mangle_identifier(ns) + remaining
    else:
        # non-nested
        remaining = mangled[2:]
        head, tail = _split_mangled_ident(remaining)
        ret = PREFIX + 'N' + mangle_identifier(ns) + head + 'E' + tail

    return ret


def _split_mangled_ident(mangled):
    """
    Returns `(head, tail)` where `head` is the `<len> + <name>` encoded
    identifier and `tail` is the remaining.
    """
    ct = int(mangled)
    ctlen = len(str(ct))
    at = ctlen + ct
    return mangled[:at], mangled[at:]


