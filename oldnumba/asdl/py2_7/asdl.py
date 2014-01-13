# -*- coding: utf-8 -*-
"""An implementation of the Zephyr Abstract Syntax Definition Language.

See http://asdl.sourceforge.net/ and
http://www.cs.princeton.edu/research/techreps/TR-554-97

Only supports top level module decl, not view.  I'm guessing that view
is intended to support the browser and I'm not interested in the
browser.

Changes for Python: Add support for module versions
"""
from __future__ import print_function, division, absolute_import

import os
import traceback

from . import spark

class Token(object):
    # spark seems to dispatch in the parser based on a token's
    # type attribute
    def __init__(self, type, lineno):
        self.type = type
        self.lineno = lineno

    def __str__(self):
        return self.type

    def __repr__(self):
        return str(self)

class Id(Token):
    def __init__(self, value, lineno):
        self.type = 'Id'
        self.value = value
        self.lineno = lineno

    def __str__(self):
        return self.value

class String(Token):
    def __init__(self, value, lineno):
        self.type = 'String'
        self.value = value
        self.lineno = lineno

class ASDLSyntaxError(Exception):

    def __init__(self, lineno, token=None, msg=None):
        self.lineno = lineno
        self.token = token
        self.msg = msg

    def __str__(self):
        if self.msg is None:
            return "Error at '%s', line %d" % (self.token, self.lineno)
        else:
            return "%s, line %d" % (self.msg, self.lineno)

class ASDLScanner(spark.GenericScanner, object):

    def tokenize(self, input):
        self.rv = []
        self.lineno = 1
        super(ASDLScanner, self).tokenize(input)
        return self.rv

    def t_id(self, s):
        r"[\w\.]+"
        # XXX doesn't distinguish upper vs. lower, which is
        # significant for ASDL.
        self.rv.append(Id(s, self.lineno))

    def t_string(self, s):
        r'"[^"]*"'
        self.rv.append(String(s, self.lineno))

    def t_xxx(self, s): # not sure what this production means
        r"<="
        self.rv.append(Token(s, self.lineno))

    def t_punctuation(self, s):
        r"[\{\}\*\=\|\(\)\,\?\:]"
        self.rv.append(Token(s, self.lineno))

    def t_comment(self, s):
        r"\-\-[^\n]*"
        pass

    def t_newline(self, s):
        r"\n"
        self.lineno += 1

    def t_whitespace(self, s):
        r"[ \t]+"
        pass

    def t_default(self, s):
        r" . +"
        raise ValueError("unmatched input: %r" % s)

class ASDLParser(spark.GenericParser, object):
    def __init__(self):
        super(ASDLParser, self).__init__("module")

    def typestring(self, tok):
        return tok.type

    def error(self, tok):
        raise ASDLSyntaxError(tok.lineno, tok)

    def p_module_0(self, xxx_todo_changeme):
        " module ::= Id Id version { } "
        (module, name, version, _0, _1) = xxx_todo_changeme
        if module.value != "module":
            raise ASDLSyntaxError(module.lineno,
                                  msg="expected 'module', found %s" % module)
        return Module(name, None, version)

    def p_module(self, xxx_todo_changeme1):
        " module ::= Id Id version { definitions } "
        (module, name, version, _0, definitions, _1) = xxx_todo_changeme1
        if module.value != "module":
            raise ASDLSyntaxError(module.lineno,
                                  msg="expected 'module', found %s" % module)
        return Module(name, definitions, version)

    def p_version(self, xxx_todo_changeme2):
        "version ::= Id String"
        (version, V) = xxx_todo_changeme2
        if version.value != "version":
            raise ASDLSyntaxError(version.lineno,
                                msg="expected 'version', found %" % version)
        return V

    def p_definition_0(self, xxx_todo_changeme3):
        " definitions ::= definition "
        (definition,) = xxx_todo_changeme3
        return definition

    def p_definition_1(self, xxx_todo_changeme4):
        " definitions ::= definition definitions "
        (definitions, definition) = xxx_todo_changeme4
        return definitions + definition

    def p_definition(self, xxx_todo_changeme5):
        " definition ::= Id = type "
        (id, _, type) = xxx_todo_changeme5
        return [Type(id, type)]

    def p_type_0(self, xxx_todo_changeme6):
        " type ::= product "
        (product,) = xxx_todo_changeme6
        return product

    def p_type_1(self, xxx_todo_changeme7):
        " type ::= sum "
        (sum,) = xxx_todo_changeme7
        return Sum(sum)

    def p_type_2(self, xxx_todo_changeme8):
        " type ::= sum Id ( fields ) "
        (sum, id, _0, attributes, _1) = xxx_todo_changeme8
        if id.value != "attributes":
            raise ASDLSyntaxError(id.lineno,
                                  msg="expected attributes, found %s" % id)
        if attributes:
            attributes.reverse()
        return Sum(sum, attributes)

    def p_product(self, xxx_todo_changeme9):
        " product ::= ( fields ) "
        (_0, fields, _1) = xxx_todo_changeme9
        fields.reverse()
        return Product(fields)

    def p_sum_0(self, xxx_todo_changeme10):
        " sum ::= constructor "
        (constructor,) = xxx_todo_changeme10
        return [constructor]

    def p_sum_1(self, xxx_todo_changeme11):
        " sum ::= constructor | sum "
        (constructor, _, sum) = xxx_todo_changeme11
        return [constructor] + sum

    def p_sum_2(self, xxx_todo_changeme12):
        " sum ::= constructor | sum "
        (constructor, _, sum) = xxx_todo_changeme12
        return [constructor] + sum

    def p_constructor_0(self, xxx_todo_changeme13):
        " constructor ::= Id "
        (id,) = xxx_todo_changeme13
        return Constructor(id)

    def p_constructor_1(self, xxx_todo_changeme14):
        " constructor ::= Id ( fields ) "
        (id, _0, fields, _1) = xxx_todo_changeme14
        fields.reverse()
        return Constructor(id, fields)

    def p_fields_0(self, xxx_todo_changeme15):
        " fields ::= field "
        (field,) = xxx_todo_changeme15
        return [field]

    def p_fields_1(self, xxx_todo_changeme16):
        " fields ::= field , fields "
        (field, _, fields) = xxx_todo_changeme16
        return fields + [field]

    def p_field_0(self, xxx_todo_changeme17):
        " field ::= Id "
        (type,) = xxx_todo_changeme17
        return Field(type)

    def p_field_1(self, xxx_todo_changeme18):
        " field ::= Id Id "
        (type, name) = xxx_todo_changeme18
        return Field(type, name)

    def p_field_2(self, xxx_todo_changeme19):
        " field ::= Id * Id "
        (type, _, name) = xxx_todo_changeme19
        return Field(type, name, seq=True)

    def p_field_3(self, xxx_todo_changeme20):
        " field ::= Id ? Id "
        (type, _, name) = xxx_todo_changeme20
        return Field(type, name, opt=True)

    def p_field_4(self, xxx_todo_changeme21):
        " field ::= Id * "
        (type, _) = xxx_todo_changeme21
        return Field(type, seq=True)

    def p_field_5(self, xxx_todo_changeme22):
        " field ::= Id ? "
        (type, _) = xxx_todo_changeme22
        return Field(type, opt=True)

builtin_types = ("identifier", "string", "int", "bool", "object")

# below is a collection of classes to capture the AST of an AST :-)
# not sure if any of the methods are useful yet, but I'm adding them
# piecemeal as they seem helpful

class AST(object):
    pass # a marker class

class Module(AST):
    def __init__(self, name, dfns, version):
        self.name = name
        self.dfns = dfns
        self.version = version
        self.types = {} # maps type name to value (from dfns)
        for type in dfns:
            self.types[type.name.value] = type.value

    def __repr__(self):
        return "Module(%s, %s)" % (self.name, self.dfns)

class Type(AST):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "Type(%s, %s)" % (self.name, self.value)

class Constructor(AST):
    def __init__(self, name, fields=None):
        self.name = name
        self.fields = fields or []

    def __repr__(self):
        return "Constructor(%s, %s)" % (self.name, self.fields)

class Field(AST):
    def __init__(self, type, name=None, seq=False, opt=False):
        self.type = type
        self.name = name
        self.seq = seq
        self.opt = opt

    def __repr__(self):
        if self.seq:
            extra = ", seq=True"
        elif self.opt:
            extra = ", opt=True"
        else:
            extra = ""
        if self.name is None:
            return "Field(%s%s)" % (self.type, extra)
        else:
            return "Field(%s, %s%s)" % (self.type, self.name, extra)

class Sum(AST):
    def __init__(self, types, attributes=None):
        self.types = types
        self.attributes = attributes or []

    def __repr__(self):
        if self.attributes is None:
            return "Sum(%s)" % self.types
        else:
            return "Sum(%s, %s)" % (self.types, self.attributes)

class Product(AST):
    def __init__(self, fields):
        self.fields = fields

    def __repr__(self):
        return "Product(%s)" % self.fields

class VisitorBase(object):

    def __init__(self, skip=False):
        self.cache = {}
        self.skip = skip

    def visit(self, object, *args):
        meth = self._dispatch(object)
        if meth is None:
            return
        try:
            meth(object, *args)
        except Exception as err:
            print(("Error visiting", repr(object)))
            print(err)
            traceback.print_exc()
            # XXX hack
            if hasattr(self, 'file'):
                self.file.flush()
            os._exit(1)

    def _dispatch(self, object):
        assert isinstance(object, AST), repr(object)
        klass = object.__class__
        meth = self.cache.get(klass)
        if meth is None:
            methname = "visit" + klass.__name__
            if self.skip:
                meth = getattr(self, methname, None)
            else:
                meth = getattr(self, methname)
            self.cache[klass] = meth
        return meth

class Check(VisitorBase):

    def __init__(self):
        super(Check, self).__init__(skip=True)
        self.cons = {}
        self.errors = 0
        self.types = {}

    def visitModule(self, mod):
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, type):
        self.visit(type.value, str(type.name))

    def visitSum(self, sum, name):
        for t in sum.types:
            self.visit(t, name)

    def visitConstructor(self, cons, name):
        key = str(cons.name)
        conflict = self.cons.get(key)
        if conflict is None:
            self.cons[key] = name
        else:
            print(("Redefinition of constructor %s" % key))
            print(("Defined in %s and %s" % (conflict, name)))
            self.errors += 1
        for f in cons.fields:
            self.visit(f, key)

    def visitField(self, field, name):
        key = str(field.type)
        l = self.types.setdefault(key, [])
        l.append(name)

    def visitProduct(self, prod, name):
        for f in prod.fields:
            self.visit(f, name)

def check(mod):
    v = Check()
    v.visit(mod)

    for t in v.types:
        if t not in mod.types and not t in builtin_types:
            v.errors += 1
            uses = ", ".join(v.types[t])
            print(("Undefined type %s, used in %s" % (t, uses)))

    return not v.errors

def parse(file):
    scanner = ASDLScanner()
    parser = ASDLParser()

    buf = open(file).read()
    tokens = scanner.tokenize(buf)
    try:
        return parser.parse(tokens)
    except ASDLSyntaxError as err:
        print(err)
        lines = buf.split("\n")
        print((lines[err.lineno - 1])) # lines starts at 0, files at 1

if __name__ == "__main__":
    import glob
    import sys

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        testdir = "tests"
        files = glob.glob(testdir + "/*.asdl")

    for file in files:
        print(file)
        mod = parse(file)
        print(("module", mod.name))
        print((len(mod.dfns), "definitions"))
        if not check(mod):
            print("Check failed")
        else:
            for dfn in mod.dfns:
                print((dfn.type))
