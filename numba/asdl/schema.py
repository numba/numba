import sys
import ast
import asdl
from collections import defaultdict, namedtuple
import contextlib

def load(name):
    '''Load a ASDL Schema by name; e.g. "Python.asdl".
        
    Returns a Schema object.

    This tries to load from the version-specific directory, first.
    If it failed, it loads from the common-directory.
    '''
    python_asdl = asdl.load(name)
    schblr = SchemaBuilder()
    schblr.visit(python_asdl)
    return schblr.schema

class _rule(namedtuple('rule', ['kind', 'fields'])):
    __slots__ = ()

    SUM = 0
    PROD = 1

    @classmethod
    def sum(cls, fields):
        return cls(kind=cls.SUM, fields=fields)

    @classmethod
    def product(cls, fields):
        return cls(kind=cls.PROD, fields=fields)

    @property
    def is_sum(self):
        return self.kind == self.SUM

    @property
    def is_product(self):
        return self.kind == self.PROD

class _debuginfo_t(namedtuple("_debuginfo_t", ['node', 'field', 'offset'])):
    __slots__ = ()

    def __str__(self):
        '''Create string reprentation to be used in SchemaError
        '''
        if self.field is not None:
            if self.offset is not None:
                return "At %s.%s[%d]" % (self.node, self.field, self.offset)
            else:
                return "At %s.%s" % (self.node, self.field)
        else:
            return "At %s" % (self.node)

def _debuginfo(node, field=None, offset=None):
    assert field or not offset
    return _debuginfo_t(node=node, field=field, offset=offset)

class SchemaError(Exception):
    def __init__(self, ctxt, msg):
        super(SchemaError, self).__init__("%s: %s" % (ctxt, msg))

class Schema(object):
    '''A Schema object that is used to verify against an AST.
    
    It is built from SchemaBuilder
        
    Usage:

        schema.verify(ast)
        schema.verify(ast, context=SchemaContext())
    '''
    def __init__(self, name):
        # name of the asdl module
        self.name = name
        # a dictionary of {type name -> fields}
        self.types = defaultdict(list)
        # a dictionary of {definition name -> _rule}
        self.dfns = {}

    def verify(self, ast, context=None):
        '''Check against an AST raises SchemaError upon error.
        
        ast --- The ast being verified
        context --- [optional] a SchemaContext.
        '''
        context = context if context is not None else SchemaContext()
        return SchemaVerifier(self, context).visit(ast)

    def debug(self):
        print "Schema %s" % self.name
        for k, fs in self.types.items():
            print '    --', k, fs
        for k, fs in self.dfns.items():
            if fs.is_sum():
                print '   ', k
                for f in fs.fields:
                    print '       |', f
            else:
                print '   ', k, '=', ', '.join(map(str, fs.fields))

class SchemaContext(object):
    '''Keep information about context:
        - builtin type handlers
        
    User may expand the builtin type handlers.  See `builtin_handlers`.
        
    '''
    def __init__(self):
        self.__builtins = {}
        self._add_default_handler()

    def _add_default_handler(self):
        self.builtin_handlers['identifier'] = _verify_identifier
        self.builtin_handlers['int'] = _verify_int
        self.builtin_handlers['string'] = _verify_string
        self.builtin_handlers['object'] = _verify_object
        self.builtin_handlers['bool'] = _verify_bool

    @property
    def builtin_handlers(self):
        '''A dictionary of type name -> handler
            
        A handler is just a callable like the following:
            
            def handler(value):
                return is_value_valid(value) # returns boolean
        '''
        return self.__builtins


class SchemaVerifier(ast.NodeVisitor):
    '''A internal class that implement a the verification logic.
    '''
    def __init__(self, schema, context):
        '''
        schema --- a Schema object that defines a valid AST.
        context --- a SchemaConctext object.
        '''
        self.schema = schema
        self.context = context
        self._debug_context = None
    
    def visit(self, node):
        '''Start verification at the node.
        
        Verification can begin at any AST node.  Can it will recursively
        verify each and every subtree.
        '''
        current = self._get_type(node)
        self._visit(node, current)

    def _visit(self, node, current):
        nodename = type(node).__name__
        # traverse the children

        for field in current:
            with self._new_debug_context(node=nodename, field=field.name):
                value = getattr(node, str(field.name), None)
                if getattr(field, 'seq', False):
                    # is sequence?
                    children = getattr(node, str(field.name), None)
                    if children is None:
                        raise SchemaError(self._debug_context, "Missing field")
                    elif not _is_iterable(children):
                        raise SchemaError(self._debug_context,
                                          "Field must be iterable")

                    for offset, child in enumerate(children):
                        with self._new_debug_context(node=nodename,
                                                     field=field.name,
                                                     offset=offset):
                            self._sentry_dfn(field.type, child)

                elif value is None:
                    if not getattr(field, 'opt', False):
                        raise SchemaError(self._debug_context, "Missing field")
                    else:
                        pass
                elif value is not None:
                    self._sentry_dfn(field.type, value)

                else:
                    assert False, field

    @contextlib.contextmanager
    def _new_debug_context(self, **kws):
        # push
        old = self._debug_context
        self._debug_context = _debuginfo(**kws)
        yield
        # pop
        self._debug_context = old

    def _sentry_child(self, child, subtypes):
        childname = type(child).__name__
        if childname not in subtypes:
            raise SchemaError(self._debug_context, "Cannot be a %s" % childname)

    def _sentry_dfn(self, name, value):
        name = str(name)

        if name in self.context.builtin_handlers:
            # is a builtin type?
            handler = self.context.builtin_handlers[name]
            if not handler(value):
                raise SchemaError(self._debug_context,
                                  "Expected %s but got %s" % \
                                  (name, type(value)))
        else:
            # not a builtin type?
            rule = self._get_dfn(name)
            if rule.is_sum:
                self._sentry_child(value, rule.fields)
                self.visit(value)
            else:
                assert rule.is_product
                name0 = type(value).__name__
                if name0 != name:
                    raise SchemaError(self._debug_context,
                                      "Expecting %s but got %s" % \
                                      (name, name0))

                self._visit(value, rule.fields)

    def _get_dfn(self, name):
        ret = self.schema.dfns.get(str(name))
        if ret is None:
            raise SchemaError(self._debug_context,
                              "Missing definition for '%s' in the ASDL" % name)
        return ret

    def _get_type(self, cls_or_name):
        name = (cls_or_name
                if isinstance(cls_or_name, str)
                else type(cls_or_name).__name__)
        ret = self.schema.types.get(name)
        if ret is None:
            raise SchemaError(self._debug_context,
                              "Unknown AST node type: %s" % name)
        return ret

class SchemaBuilder(asdl.VisitorBase):
    '''A single instance of SchemaBuilder can be used build different 
    Schema from different ASDL.

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
        self.schema.dfns[str(name)] = _rule.sum([str(t.name)
                                                 for t in sum.types])
        for t in sum.types:
            self.visit(t, name, sum.attributes)

    def visitConstructor(self, cons, name, attr=[]):
        typename = str(cons.name)
        fields = self.schema.types[typename]
        for f in cons.fields:
            fields.append(f)
        for f in attr:
            fields.append(f)


    def visitField(self, field, name):
        key = str(field.type)

    def visitProduct(self, prod, name):
        assert not hasattr(prod, 'attributes')
        self.schema.dfns[str(name)] = _rule.product(prod.fields)
        for f in prod.fields:
            self.visit(f, name)

    @property
    def schema(self):
        return self.__schema

#
# Builtin types handler
#

def _verify_identifier(value):
    return isinstance(value, str)

def _verify_int(value):
    return isinstance(value, int) or isinstance(value, long)

def _verify_string(value):
    return isinstance(value, str)

def _verify_object(value):
    return isinstance(value, object)

def _verify_bool(value):
    return isinstance(value, bool)

#
#  Utilities
#
def _is_iterable(value):
    try:
        iter(value)
    except TypeError:
        return False
    else:
        return True
