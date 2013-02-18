import sys
import opcode
import ast
import pprint

import numba
from .minivect.complex_support import Complex64, Complex128, Complex256
from .minivect import miniast, minitypes
from numba import typesystem
from numba.typesystem.typemapper import NumbaTypeMapper

def itercode(code):
    """Return a generator of byte-offset, opcode, and argument
    from a byte-code-string
    """
    i = 0
    extended_arg = 0
    n = len(code)
    while i < n:
        c = code[i]
        num = i
        op = ord(c)
        i = i + 1
        oparg = None
        if op >= opcode.HAVE_ARGUMENT:
            oparg = ord(code[i]) + ord(code[i + 1]) * 256 + extended_arg
            extended_arg = 0
            i = i + 2
            if op == opcode.EXTENDED_ARG:
                extended_arg = oparg * 65536L

        delta = yield num, op, oparg
        if delta is not None:
            abs_rel, dst = delta
            assert abs_rel == 'abs' or abs_rel == 'rel'
            i = dst if abs_rel == 'abs' else i + dst

def debugout(*args):
    '''This is a magic function.  If you use it in compiled functions,
    Numba should generate code for displaying the received value.'''
    if __debug__:
        print("debugout (non-translated): %s" % (''.join((str(arg)
                                                          for arg in args)),))

def process_signature(sigstr, name=None):
    '''
    Given a signature string consisting of a return type, argument
    types, and possibly a function name, return a signature object.
    '''
    sigstr = sigstr.replace('*', '.pointer()')
    parts = sigstr.split()
    types_dict = dict(numba.__dict__, d=numba.double, i=numba.int_)
    loc = {}
    # FIXME:  Need something more robust to differentiate between
    #   name ret(arg1,arg2)
    #   and ret(arg1, arg2) or ret ( arg1, arg2 )
    if len(parts) < 2 or '(' in parts[0] or '[' in parts[0] or '('==parts[1][0]:
        signature = eval(sigstr, loc, types_dict)
        signature.name = None
    else: # Signature has a name
        signature = eval(' '.join(parts[1:]), loc, types_dict)
        signature.name = parts[0]
    if name is not None:
        signature.name = name
    return signature

class NumbaContext(miniast.LLVMContext):
    # debug = True
    # debug_elements = True

    # Accept dynamic arguments
    astbuilder_cls = miniast.DynamicArgumentASTBuilder

    shape_type = minitypes.npy_intp.pointer()
    strides_type = shape_type

    def init(self):
        self.astbuilder = self.astbuilder_cls(self)
        self.typemapper = NumbaTypeMapper(self)

    def is_object(self, type):
        return super(NumbaContext, self).is_object(type) or type.is_array

    def promote_types(self, *args, **kwargs):
        return self.typemapper.promote_types(*args, **kwargs)

def get_minivect_context():
    return NumbaContext()

context = get_minivect_context()

def ast2tree (node, include_attrs = True):
    def _transform(node):
        if isinstance(node, ast.AST):
            fields = ((a, _transform(b))
                      for a, b in ast.iter_fields(node))
            if include_attrs:
                attrs = ((a, _transform(getattr(node, a)))
                         for a in node._attributes
                         if hasattr(node, a))
                return (node.__class__.__name__, dict(fields), dict(attrs))
            return (node.__class__.__name__, dict(fields))
        elif isinstance(node, list):
            return [_transform(x) for x in node]
        return node
    if not isinstance(node, ast.AST):
        raise TypeError('expected AST, got %r' % node.__class__.__name__)
    return _transform(node)

def pformat_ast (node, include_attrs = True, **kws):
    return pprint.pformat(ast2tree(node, include_attrs), **kws)

def dump(node):
    print pformat_ast(node)

class TypedProperty(object):
    '''Defines a class property that does a type check in the setter.'''

    def __new__(cls, ty, doc, default=None):
        rv = super(TypedProperty, cls).__new__(cls)
        cls.__init__(rv, ty, doc, default)
        return property(rv.getter, rv.setter, rv.deleter, doc)

    def __init__(self, ty, doc, default=None):
        self.propname = '_numba_property_%d' % (id(self),)
        self.default = default
        self.ty = ty
        self.doc = doc

    def getter(self, obj):
        return getattr(obj, self.propname, self.default)

    def setter(self, obj, new_val):
        if not isinstance(new_val, self.ty):
            raise ValueError(
                'Invalid property setting, expected instance of type(s) %r '
                '(got %r).' % (self.ty, type(new_val)))
        setattr(obj, self.propname, new_val)

    def deleter(self, obj):
        delattr(obj, self.propname)

class WriteOnceTypedProperty(TypedProperty):
    def __init__(self, ty, doc, default=None):
        super(WriteOnceTypedProperty, self).__init__(ty, doc, default)

    def setter(self, obj, *args, **kws):
        assert not hasattr(obj, self.propname)
        return super(WriteOnceTypedProperty, self).setter(obj, *args, **kws)

#------------------------------------------------------------------------
# File Opening Utilities
#------------------------------------------------------------------------

# file name encodings (function copied from Cython)

def decode_filename(filename):
    if isinstance(filename, unicode):
        return filename
    try:
        filename_encoding = sys.getfilesystemencoding()
        if filename_encoding is None:
            filename_encoding = sys.getdefaultencoding()
        filename = filename.decode(filename_encoding)
    except UnicodeDecodeError:
        pass
    return filename
