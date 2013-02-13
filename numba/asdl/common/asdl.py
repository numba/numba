"""An implementation of the Zephyr Abstract Syntax Definition Language.

See http://asdl.sourceforge.net/ and
http://www.cs.princeton.edu/research/techreps/TR-554-97

Only supports top level module decl, not view.  I'm guessing that view
is intended to support the browser and I'm not interested in the
browser.

Changes for Python: Add support for module versions
"""

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

    def p_module_0(self, (module, name, version, _0, _1)):
        " module ::= Id Id version { } "
        if module.value != "module":
            raise ASDLSyntaxError(module.lineno,
                                  msg="expected 'module', found %s" % module)
        return Module(name, None, version)

    def p_module(self, (module, name, version, _0, definitions, _1)):
        " module ::= Id Id version { definitions } "
        if module.value != "module":
            raise ASDLSyntaxError(module.lineno,
                                  msg="expected 'module', found %s" % module)
        return Module(name, definitions, version)

    def p_version(self, (version, V)):
        "version ::= Id String"
        if version.value != "version":
            raise ASDLSyntaxError(version.lineno,
                                msg="expected 'version', found %" % version)
        return V

    def p_definition_0(self, (definition,)):
        " definitions ::= definition "
        return definition

    def p_definition_1(self, (definitions, definition)):
        " definitions ::= definition definitions "
        return definitions + definition

    def p_definition(self, (id, _, type)):
        " definition ::= Id = type "
        return [Type(id, type)]

    def p_type_0(self, (product,)):
        " type ::= product "
        return product

    def p_type_1(self, (sum,)):
        " type ::= sum "
        return Sum(sum)

    def p_type_2(self, (sum, id, _0, attributes, _1)):
        " type ::= sum Id ( fields ) "
        if id.value != "attributes":
            raise ASDLSyntaxError(id.lineno,
                                  msg="expected attributes, found %s" % id)
        if attributes:
            attributes.reverse()
        return Sum(sum, attributes)

    def p_product(self, (_0, fields, _1)):
        " product ::= ( fields ) "
        # XXX can't I just construct things in the right order?
        fields.reverse()
        return Product(fields)

    def p_sum_0(self, (constructor,)):
        " sum ::= constructor "
        return [constructor]

    def p_sum_1(self, (constructor, _, sum)):
        " sum ::= constructor | sum "
        return [constructor] + sum

    def p_sum_2(self, (constructor, _, sum)):
        " sum ::= constructor | sum "
        return [constructor] + sum

    def p_constructor_0(self, (id,)):
        " constructor ::= Id "
        return Constructor(id)

    def p_constructor_1(self, (id, _0, fields, _1)):
        " constructor ::= Id ( fields ) "
        # XXX can't I just construct things in the right order?
        fields.reverse()
        return Constructor(id, fields)

    def p_fields_0(self, (field,)):
        " fields ::= field "
        return [field]

    def p_fields_1(self, (field, _, fields)):
        " fields ::= field , fields "
        return fields + [field]

    def p_field_0(self, (type,)):
        " field ::= Id "
        return Field(type)

    def p_field_1(self, (type, name)):
        " field ::= Id Id "
        return Field(type, name)

    def p_field_2(self, (type, _, name)):
        " field ::= Id * Id "
        return Field(type, name, seq=True)

    def p_field_3(self, (type, _, name)):
        " field ::= Id ? Id "
        return Field(type, name, opt=True)

    def p_field_4(self, (type, _)):
        " field ::= Id * "
        return Field(type, seq=True)

    def p_field_5(self, (type, _)):
        " field ::= Id ? "
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
            print "Error visiting", repr(object)
            print err
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
            print "Redefinition of constructor %s" % key
            print "Defined in %s and %s" % (conflict, name)
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
            print "Undefined type %s, used in %s" % (t, uses)

    return not v.errors

def parse(file):
    scanner = ASDLScanner()
    parser = ASDLParser()

    buf = open(file).read()
    tokens = scanner.tokenize(buf)
    try:
        return parser.parse(tokens)
    except ASDLSyntaxError as err:
        print err
        lines = buf.split("\n")
        print lines[err.lineno - 1] # lines starts at 0, files at 1

if __name__ == "__main__":
    import glob
    import sys

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        testdir = "tests"
        files = glob.glob(testdir + "/*.asdl")

    for file in files:
        print file
        mod = parse(file)
        print "module", mod.name
        print len(mod.dfns), "definitions"
        if not check(mod):
            print "Check failed"
        else:
            for dfn in mod.dfns:
                print dfn.type
