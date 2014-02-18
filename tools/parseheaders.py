

from __future__ import print_function, absolute_import, division
import re
from collections import namedtuple

try:
    xrange
except NameError:
    xrange = range


re_white = re.compile(r"[ \n\r]+", re.MULTILINE)
re_atom = re.compile(r"[_a-z][_a-z0-9]*", re.I)
re_open_paren = re.compile(r"\(")
re_close_paren = re.compile(r"\)")
re_operators = re.compile(r"[+\-*/]")
re_end = re.compile(r";")
re_comma = re.compile(r",")

PATTERNS = [('atom', re_atom),
            ('white', re_white),
            ('open_paren', re_open_paren),
            ('close_paren', re_close_paren),
            ('operator', re_operators),
            ('end', re_end),
            ('comma', re_comma),]


Token = namedtuple("Token", ["text", "kind"])


class EOS(StopIteration):
    pass


class Tokenizer(object):
    def __init__(self, content):
        self.content = content
        self.pos = 0

    def next(self):
        while self.head:
            # Skip comments
            if self.head[:2] == '/*':
                self.pos += 2
                self.until_end_of_mt_comment()
                continue
            # Patterns
            for name, pat in PATTERNS:
                tok = self.try_match(pat)
                if tok:
                    return Token(text=tok, kind=name)
            raise ValueError("Unknown string pattern: %r", self.head[:30])
        raise EOS

    def until_end_of_mt_comment(self):
        while self.head[:2] != '*/':
            self.pos += 1
        self.pos += 2

    @property
    def head(self):
        return self.content[self.pos + 1:]

    def try_match(self, regex):
        m = regex.match(self.head)
        if not m:
            return
        got = m.group(0)
        self.pos += len(got)
        return got


class Declaration(object):
    __slots__ = 'name', 'return_type', 'args'
    def __init__(self, name, return_type, args):
        self.name = name
        self.return_type = return_type
        self.args = tuple(args)

    def __repr__(self):
        return "Declaration(%s, %s, %s)" % (self.name, self.return_type,
                                            self.args)


class DeclParser(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.buffer = []
        self.declarations = []

    def next_token(self):
        while True:
            try:
                tok = self.tokenizer.next()
            except EOS:
                tok = Token(text=None, kind='EOS')

            if tok.kind != 'white':
                return tok

    def peek(self, n=0):
        while len(self.buffer) <= n:
            self.buffer.append(self.next_token())
        return self.buffer[n]

    def one(self):
        tok = self.peek()
        self.buffer = self.buffer[1:]
        return tok

    def many(self, n):
        toks = self.peek(n)
        self.buffer = self.buffer[n:]
        return toks

    def parse_decl(self):
        return_type = self.one()
        assert return_type.kind == 'atom'
        name = self.one()
        # ignore (
        self.one()

        args = []
        while self.peek().kind != 'close_paren':
            arginfo = []
            while self.peek().kind not in ('close_paren', 'comma'):
                arginfo.append(self.one())
            args.append(tuple(arginfo))
            if self.peek().kind != 'close_paren':
                comma = self.one()
                assert comma.kind == 'comma'


        close = self.one()
        assert close.kind == 'close_paren', close.kind

        end = self.one()
        assert end.kind == 'end', end.kind

        return Declaration(name, return_type, args)

    def parse(self):
        while self.peek().kind != 'EOS':
            decl = self.parse_decl()
            self.declarations.append(decl)

decl_template = '''
%s = (%r, (%s,))
'''

arg_template = "(%r, %r)"


def format_arg(arginfo):
    typeinfo = [info.text for info in arginfo[:-1] if info.text != 'const']
    typ = ''.join(typeinfo)
    string = arg_template % (arginfo[-1].text, typ)
    return string


def format_declaration(decl):
    args = ', '.join(format_arg(arg) for arg in decl.args)
    string = decl_template % (decl.name.text, decl.return_type.text, args)
    return string


def main(filename):
    with open(filename) as srcfile:
        content = srcfile.read()
        parser = DeclParser(Tokenizer(content))
        parser.parse()
        for decl in parser.declarations:
            print(format_declaration(decl))

if __name__ == '__main__':
    main('/Users/sklam/dev/numbapro/cuda_includes/cusparse.h')
