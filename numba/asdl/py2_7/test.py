import ast
import asdl
from collections import defaultdict, namedtuple
import contextlib

SUM = 'sum'
PROD = 'prod'

Rule = namedtuple('Rule', ['kind', 'fields'])

class SchemaError(Exception):
    pass

class Schema(object):
    def __init__(self, name):
        # name of the asdl module
        self.name = name
        # a dictionary of {type name -> fields}
        self.types = defaultdict(list)
        # a dictionary of {definition name -> (['sum'|'product'], fields)}
        self.dfns = {}

    def verify(self, ast):
        '''Check against an AST
        '''
        return SchemaVerifier(self).visit(ast)

    def debug(self):
        print "Schema %s" % self.name
        for k, fs in self.types.items():
            print '    --', k, fs
        for k, fs in self.dfns.items():
            if fs.kind == SUM:
                print '   ', k
                for f in fs.fields:
                    print '       |', f
            else:
                print '   ', k, '=', ', '.join(map(str, fs.fields))

asdl_builtins = 'identifier', 'int', 'string', 'object', 'bool'

class SchemaVerifier(ast.NodeVisitor):
    def __init__(self, schema):
        self.schema = schema
    
    def visit(self, node):
        current = self._get_type(node)
        self._visit(node, current)

    def _visit(self, node, current):
        nodename = type(node).__name__
        # traverse the children

        for field in current:
            value = getattr(node, str(field.name), None)
            if getattr(field, 'seq', False):
                # is sequence?
                children = getattr(node, str(field.name), None)
                if children is None:
                    raise SchemaError("Missing field '%s' in %s" % (field.name,
                                                                    nodename))

                for child in children:
                    self._sentry_dfn(field.type, child)

            elif value is None:
                if not getattr(field, 'opt', False):
                    raise SchemaError("Missing field '%s' in %s" % (field.name,
                                                                    nodename))
                else:
                    pass
            elif value is not None:
                self._sentry_dfn(field.type, value)

            else:
                assert False, field

    def _sentry_child(self, child, subtypes):
        childname = type(child).__name__
        if childname not in subtypes:
            raise SchemaError("Cannot be a %s" % childname)

    def _sentry_dfn(self, name, value):
        name = str(name)

        if name in asdl_builtins:
            if name == 'int':
                assert isinstance(value, int)
            elif name == 'object':
                assert isinstance(value, object)
            elif name == 'identifier':
                assert isinstance(value, str)
            else:
                assert False, (name, value)
        else:
            kind, subtypes = self._get_dfn(name)
            if kind == SUM:
                self._sentry_child(value, subtypes)
                self.visit(value)
            else:
                assert kind == PROD
                name0 = type(value).__name__
                if name0 != name:
                    raise SchemaError("Expecting %s but got %s" % \
                                      (name, name0))

                self._visit(value, subtypes)

    def _get_dfn(self, name):
        ret = self.schema.dfns.get(str(name))
        if ret is None:
            raise SchemaError("Missing definition for '%s' in the ASDL" % name)
        return ret

    def _get_type(self, cls_or_name):
        name = (cls_or_name
                if isinstance(cls_or_name, str)
                else type(cls_or_name).__name__)
        ret = self.schema.types.get(name)
        if ret is None:
            raise SchemaError("Unknown AST node type: %s" % name)
        return ret

class SchemaBuilder(asdl.VisitorBase):
    '''
    Usage:
        schblr = SchemaBuilder()
        schblr.visit(some_asdl)
        schema = schblr.schema      # get the schema object
        
    NOTE:
        - It ignore the attributes of the nodes.
    '''
    def __init__(self):
        super(SchemaBuilder, self).__init__()

    def visitModule(self, mod):
        self.__schema = Schema(str(mod.name))
        for dfn in mod.dfns:
            self.visit(dfn)

    def visitType(self, type):
        self.visit(type.value, str(type.name))

    def visitSum(self, sum, name):
        self.schema.dfns[str(name)] = Rule(SUM, [str(t.name)
                                                 for t in sum.types])
        for t in sum.types:
            self.visit(t, name)

    def visitConstructor(self, cons, name):
        typename = str(cons.name)
        fields = self.schema.types[typename]
        for f in cons.fields:
            fields.append(f)

    def visitField(self, field, name):
        key = str(field.type)

    def visitProduct(self, prod, name):
        self.schema.dfns[str(name)] = Rule(PROD, prod.fields)
        for f in prod.fields:
            self.visit(f, name)

    @property
    def schema(self):
        return self.__schema

def main():
    srcfile = 'Python.asdl'

    python_asdl = asdl.parse(srcfile)
    assert asdl.check(python_asdl)

    schblr = SchemaBuilder()
    schblr.visit(python_asdl)
    schema = schblr.schema
    schema.debug()


    the_ast = ast.parse('''
def foo(x, y):
    return 1 + 2
        ''')
    schema.verify(the_ast)

    print 'ok'

if __name__ == '__main__':
    main()