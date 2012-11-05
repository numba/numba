import ast
import ctypes
import collections

import numba
import numba.functions
from numba import *
from .symtab import Variable
from . import _numba_types as numba_types
from numba import utils, translate, error
from numba.minivect import minitypes

import llvm.core

from ndarray_helpers import PyArrayAccessor

context = utils.get_minivect_context()

#
### Convenience functions
#

def _const_int(X):
    return llvm.core.Constant.int(llvm.core.Type.int(), X)

def objconst(obj):
    return const(obj, object_)

def const(obj, type):
    if numba_types.is_obj(type):
        node = ObjectInjectNode(obj, type)
    else:
        node = ConstNode(obj, type)

    return node

def call_pyfunc(py_func, args):
    "Generate an object call for a python function given during compilation time"
    func = ObjectInjectNode(py_func)
    return ObjectCallNode(None, func, args)

def index(node, constant_index, load=True, type=int_):
    if load:
        ctx = ast.Load()
    else:
        ctx = ast.Store()

    index = ast.Index(ConstNode(constant_index, type))
    index.type = type
    index.variable = Variable(type)
    return ast.Subscript(value=node, slice=index, ctx=ctx)

def ptrtoint(node):
    return CoercionNode(node, Py_uintptr_t)

def ptrfromint(intval, dst_ptr_type):
    return CoercionNode(ConstNode(intval, Py_uintptr_t), dst_ptr_type)

printing = False

def inject_print(function_cache, node):
    node = function_cache.call('PyObject_Str', node)
    node = function_cache.call('puts', node)
    return node

def print_(translator, node):
    global printing
    if printing:
        return

    printing = True

    node = inject_print(translator.function_cache, node)
    node = translator.ast.pipeline.late_specializer(node)
    translator.visit(node)

    printing = False

def print_llvm(translator, type, llvm_value):
    return print_(translator, LLVMValueRefNode(type, llvm_value))

#
### AST nodes
#

class Node(ast.AST):
    """
    Superclass for Numba AST nodes
    """
    _fields = []

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def _variable_get(self):
        if not hasattr(self, '_variable'):
            self._variable = Variable(self.type)

        return self._variable

    def _variable_set(self, variable):
        self._variable = variable

    variable = property(_variable_get, _variable_set)

    @property
    def cloneable(self):
        return CloneableNode(self)

class CoercionNode(Node):
    """
    Coerce a node to a different type
    """

    _fields = ['node']
    _attributes = ['type', 'name']

    def __new__(cls, node, dst_type, name=''):
        type = getattr(node, 'type', None) or node.variable.type
        #if type == dst_type:
        #    return node

        if isinstance(node, ConstNode) and dst_type.is_numeric:
            try:
                node.cast(dst_type)
            except TypeError:
                pass
            else:
                return node

        if dst_type.is_pointer and node.type.is_int:
            assert node.type == Py_uintptr_t

        return super(CoercionNode, cls).__new__(cls, node, dst_type, name=name)

    def __init__(self, node, dst_type, name=''):
        if node is self:
            # We are trying to coerce a CoercionNode which already has the
            # right type, so __new__ returns a CoercionNode, which then results
            # in __init__ being called
            return

        self.dst_type = dst_type
        self.type = dst_type
        self.name = name

        self.node = self.verify_conversion(dst_type, node)

        if (dst_type.is_object and not node.variable.type.is_object and
                isinstance(node, ArrayAttributeNode)):
            self.node = self.coerce_numpy_attribute(node)

    def coerce_numpy_attribute(self, node):
        """
        Numpy array attributes, such as 'data', get rewritten to direct
        accesses. Since they are being coerced back to objects, use a generic
        attribute access instead.
        """
        node = ast.Attribute(value=node.array, attr=node.attr_name,
                             ctx=ast.Load())
        node.variable = Variable(object_)
        node.type = object_
        return node

    @classmethod
    def coerce(cls, node_or_nodes, dst_type):
        if isinstance(node_or_nodes, list):
            return [cls(node, dst_type) for node in node_or_nodes]
        return cls(node_or_nodes, dst_type)

    def verify_conversion(self, dst_type, node):
        if ((node.variable.type.is_complex or dst_type.is_complex) and
            (node.variable.type.is_object or dst_type.is_object)):
            if dst_type.is_complex:
                complex_type = dst_type
            else:
                complex_type = node.variable.type

            if not complex_type == complex128:
                node = CoercionNode(node, complex128)

        return node


class CoerceToObject(CoercionNode):
    "Coerce native values to objects"

class CoerceToNative(CoercionNode):
    "Coerce objects to native values"


class DeferredCoercionNode(Node):
    """
    Coerce to the type of the given variable. The type of the variable may
    change in the meantime (e.g. may be promoted or demoted).
    """

    _fields = ['node']

    def __init__(self, node, variable):
        self.node = node
        self.variable = variable


class ConstNode(Node):
    """
    Wrap a constant.
    """

    _attributes = ['type', 'pyval']

    def __init__(self, pyval, type=None):
        if type is None:
            type = context.typemapper.from_python(pyval)

        # if pyval is not _NULL:
        #     assert not type.is_object

        self.variable = Variable(type, is_constant=True, constant_value=pyval)
        self.type = type
        self.pyval = pyval

    def value(self, translator):
        builder = translator.builder

        type = self.type
        ltype = type.to_llvm(context)

        constant = self.pyval

        if constant is _NULL:
            lvalue = llvm.core.Constant.null(type.to_llvm(context))
        elif type.is_float:
            lvalue = llvm.core.Constant.real(ltype, constant)
        elif type.is_int:
            lvalue = llvm.core.Constant.int(ltype, constant)
        elif type.is_complex:
            real = ConstNode(constant.real, type.base_type)
            imag = ConstNode(constant.imag, type.base_type)
            lvalue = llvm.core.Constant.struct([real.value(translator),
                                                imag.value(translator)])
        elif type.is_pointer:
#            lvalue = translator.visit(ptrtoint(self.pyval))
            addr_int = translator.visit(ConstNode(self.pyval, type=Py_uintptr_t))
            lvalue = translator.builder.inttoptr(addr_int, ltype)
        elif type.is_object:
            raise NotImplementedError("Use ObjectInjectNode")
        elif type.is_c_string:
            lvalue = translate._LLVMModuleUtils.get_string_constant(
                                            translator.mod, constant)
            type_char_p = numba_types.c_string_type.to_llvm(translator.context)
            lvalue = translator.builder.bitcast(lvalue, type_char_p)
        elif type.is_function:
            # lvalue = map_to_function(constant, type, self.mod)
            raise NotImplementedError
        else:
            raise NotImplementedError("Constant %s of type %s" %
                                                        (self.pyval, type))

        return lvalue

    def cast(self, dst_type):
        if dst_type.is_int:
            caster = int
        elif dst_type.is_float:
            caster = float
        elif dst_type.is_complex:
            caster = complex
        else:
            raise NotImplementedError(dst_type)

        self.pyval = caster(self.pyval)
        self.type = dst_type
        self.variable.type = dst_type

_NULL = object()
NULL_obj = ConstNode(_NULL, object_)
NULL = ConstNode(_NULL, void.pointer())

class FunctionCallNode(Node):
    _attributes = ['signature', 'type', 'name']

    def __init__(self, signature, args, name=''):
        self.signature = signature
        self.type = signature.return_type
        self.name = name
        self.original_args = args

class NativeCallNode(FunctionCallNode):

    _fields = ['args']

    def __init__(self, signature, args, llvm_func, py_func=None,
                 badval=None, goodval=None,
                 exc_type=None, exc_msg=None, exc_args=None,
                 skip_self=False, **kw):
        super(NativeCallNode, self).__init__(signature, args, **kw)
        self.llvm_func = llvm_func
        self.py_func = py_func
        self.skip_self = skip_self
        self.type = signature.return_type
        self.coerce_args()

        self.badval = badval
        self.goodval = goodval
        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.exc_args = exc_args

    def coerce_args(self):
        self.args = list(self.original_args)
        for i, dst_type in enumerate(self.signature.args[self.skip_self:]):
            arg = self.args[i]
            self.args[i] = CoercionNode(arg, dst_type,
                                        name='func_%s_arg%d' % (self.name, i))

class NativeFunctionCallNode(NativeCallNode):
    """
    Call a function which is given as a node
    """

    _fields = ['function', 'args']

    def __init__(self, signature, function_node, args, **kw):
        super(NativeFunctionCallNode, self).__init__(signature, args, None,
                                                     None, **kw)
        self.function = function_node

class LLMacroNode (NativeCallNode):
    '''
    Inject a low-level macro in the function at the call site.

    Low-level macros are Python functions that take a FunctionCache
    instance, a LLVM builder instance, and a set of arguments,
    construct LLVM code, and return some kind of LLVM value result.
    The passed signature should reflect the Numba types of the
    expected input parameters, and the type of the resulting value
    (this does not restrict polymorphism at the LLVM type level in the
    macro expansion function).
    '''

    _fields = ['macro', 'args']

    def __init__(self, signature, macro, *args, **kw):
        super(LLMacroNode, self).__init__(signature, args, None, None, **kw)
        self.macro = macro

class MathNode(Node):
    """
    Represents a high-level call to a math function.
    """

    _fields = ['arg']

    def __init__(self, py_func, signature, arg, **kwargs):
        super(MathNode, self).__init__(**kwargs)
        self.py_func = py_func
        self.signature = signature
        self.arg = arg
        self.type = signature.return_type

class LLVMExternalFunctionNode(Node):
    '''For calling an external llvm function where you only have the 
    signature and the function name.
    '''
    def __init__(self, signature, fname):
        super(LLVMExternalFunctionNode, self).__init__(signature=signature,
                                                       fname=fname)

class LLVMIntrinsicNode(NativeCallNode):
    "Call an llvm intrinsic function"

    def __init__(self, signature, args, func_name, **kw):
        super(LLVMIntrinsicNode, self).__init__(signature, args, None, **kw)
        self.func_name = func_name

class MathCallNode(NativeCallNode):
    "Low level call a libc math function"

class CTypesCallNode(NativeCallNode):
    "Call a ctypes function"

    _fields = NativeCallNode._fields + ['function']

    def __init__(self, signature, args, ctypes_func_type, py_func=None, **kw):
        super(CTypesCallNode, self).__init__(signature, args, None,
                                             py_func, **kw)
        self.pointer = ctypes.cast(py_func, ctypes.c_void_p).value
        # self.pointer = ctypes.addressof(ctypes_func_type)
        self.function = ConstNode(self.pointer, signature.pointer())


class ObjectCallNode(FunctionCallNode):
    _fields = ['function', 'args_tuple', 'kwargs_dict']

    def __init__(self, signature, func, args, keywords=None, py_func=None, **kw):
        if py_func and not kw.get('name', None):
            kw['name'] = py_func.__name__
        if signature is None:
            signature = minitypes.FunctionType(return_type=object_,
                                               args=[object_] * len(args))
            if keywords:
                signature.args.extend([object_] * len(keywords))

        super(ObjectCallNode, self).__init__(signature, args)
        assert func is not None
        self.function = func
        self.py_func = py_func

        self.args_tuple = ast.Tuple(elts=args, ctx=ast.Load())
        self.args_tuple.variable = Variable(numba_types.TupleType(
                                                size=len(args)))

        if keywords:
            keywords = [(ConstNode(k.arg), k.value) for k in keywords]
            keys, values = zip(*keywords)
            self.kwargs_dict = ast.Dict(list(keys), list(values))
            self.kwargs_dict.variable = Variable(minitypes.object_)
        else:
            self.kwargs_dict = NULL_obj

        self.type = signature.return_type

class ComplexConjugateNode(Node):
    "mycomplex.conjugate()"

    _fields = ['complex_node']

    def __init__(self, complex_node, **kwargs):
        super(ComplexConjugateNode, self).__init__(**kwargs)
        self.complex_node = complex_node

class CheckErrorNode(Node):
    """
    Check for an exception.

        badval: if this value is returned, propagate an error
        goodval: if this value is not returned, propagate an error

    If exc_type, exc_msg and optionally exc_args are given, an error is
    raised instead of propagating it.
    """

    _fields = ['return_value', 'badval', 'raise_node']

    def __init__(self, return_value, badval=None, goodval=None,
                 exc_type=None, exc_msg=None, exc_args=None,
                 **kwargs):
        super(CheckErrorNode, self).__init__(**kwargs)
        self.return_value = return_value

        if badval is not None and not isinstance(badval, ast.AST):
            badval = ConstNode(badval, return_value.type)
        if goodval is not None and not isinstance(goodval, ast.AST):
            goodval = ConstNode(goodval, return_value.type)

        self.badval = badval
        self.goodval = goodval

        self.raise_node = RaiseNode(exc_type, exc_msg, exc_args)

class RaiseNode(Node):

    _fields = ['exc_type', 'exc_msg', 'exc_args']

    def __init__(self, exc_type, exc_msg, exc_args=None, print_on_trap=True,
                 **kwargs):
        super(RaiseNode, self).__init__(**kwargs)
        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.exc_args = exc_args

        self.print_on_trap = print_on_trap

class PropagateNode(Node):
    """
    Propagate an exception (jump to the error label).
    """

class ObjectInjectNode(Node):
    """
    Refer to a Python object in the llvm code.
    """

    _attributes = ['object', 'type']

    def __init__(self, object, type=None, **kwargs):
        super(ObjectInjectNode, self).__init__(**kwargs)
        self.object = object
        self.type = type or object_
        self.variable = Variable(self.type, is_constant=True,
                                 constant_value=object)

NoneNode = ObjectInjectNode(None, object_)

class ObjectTempNode(Node):
    """
    Coerce a node to a temporary which is reference counted.
    """

    _fields = ['node']

    def __init__(self, node, incref=False):
        assert not isinstance(node, ObjectTempNode)
        self.node = node
        self.llvm_temp = None
        self.type = getattr(node, 'type', node.variable.type)
        self.incref = incref

class NoneNode(Node):
    """
    Return None.
    """

    type = numba_types.NoneType()
    variable = Variable(type)

class ObjectTempRefNode(Node):
    """
    Reference an ObjectTempNode, without evaluating its subexpressions.
    The ObjectTempNode must already have been evaluated.
    """

    _fields = []

    def __init__(self, obj_temp_node, **kwargs):
        super(ObjectTempRefNode, self).__init__(**kwargs)
        self.obj_temp_node = obj_temp_node

class CloneableNode(Node):
    """
    Create a node that can be cloned. This allows sub-expressions to be
    re-used without re-evaluating them.
    """

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(CloneableNode, self).__init__(**kwargs)
        self.node = node
        self.clone_nodes = []
        self.type = node.type

    @property
    def clone(self):
        return CloneNode(self)

class CloneNode(Node):
    """
    Clone a CloneableNode. This allows the node's sub-expressions to be
    re-used without re-evaluating them.

    The CloneableNode must be evaluated before the CloneNode is evaluated!
    """

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(CloneNode, self).__init__(**kwargs)

        assert isinstance(node, CloneableNode)
        self.node = node
        self.type = node.type
        node.clone_nodes.append(self)

        self.llvm_value = None

class ExpressionNode(Node):
    """
    Node that allows an expression to execute a bunch of statements first.
    """

    _fields = ['stmts', 'expr']

    def __init__(self, stmts, expr, **kwargs):
        super(ExpressionNode, self).__init__(**kwargs)
        self.stmts = stmts
        self.expr = expr
        self.type = expr.variable.type

class LLVMValueRefNode(Node):
    """
    Wrap an LLVM value.
    """

    _fields = []

    def __init__(self, type, llvm_value):
        self.type = type
        self.llvm_value = llvm_value


class TempNode(Node): #, ast.Name):
    """
    Create a temporary to store values in. Does not perform reference counting.
    """

    temp_counter = 0

    def __init__(self, type):
        self.type = type
        self.variable = Variable(type, name='___numba_%d' % self.temp_counter,
                                 is_local=True)
        TempNode.temp_counter += 1
        self.llvm_temp = None

    def load(self):
        return TempLoadNode(temp=self)

    def store(self):
        return TempStoreNode(temp=self)

class TempLoadNode(Node):
    _fields = ['temp']

    def __init__(self, temp):
        self.temp = temp
        self.type = temp.type
        self.variable = Variable(self.type)

class TempStoreNode(TempLoadNode):
    _fields = ['temp']

class IncrefNode(Node):

    _fields = ['value']

    def __init__(self, value, **kwargs):
        super(IncrefNode, self).__init__(**kwargs)
        self.value = value

class DecrefNode(IncrefNode):
    pass

class DataPointerNode(Node):

    _fields = ['node', 'index']

    def __init__(self, node, slice, ctx):
        self.node = node
        self.slice = slice
        self.type = node.type.dtype
        self.variable = Variable(node.type)
        self.ctx = ctx

    @property
    def ndim(self):
        return self.variable.type.ndim

    def data_descriptors(self, llvm_value, builder):
        '''
        Returns a tuple of (dptr, strides)
        - dptr:    a pointer of the data buffer
        - strides: a pointer to an array of stride information;
                   has `ndim` elements.
        '''
        acc = PyArrayAccessor(builder, llvm_value)
        return acc.data, acc.strides

    def subscript(self, translator, llvm_value, indices):
        builder = translator.builder
        caster = translator.caster
        context = translator.context

        dptr, strides = self.data_descriptors(llvm_value, builder)
#        data_ty = self.type.to_llvm(context)
#        data_ptr_ty = llvm.core.Type.pointer(data_ty)
#        ptr = builder.bitcast(dptr, data_ptr_ty)
#        return ptr

        ndim = self.ndim

        offset = _const_int(0)

        if not isinstance(indices, collections.Iterable):
            indices = (indices,)

        for i, index in zip(range(ndim), indices):
            # why is the indices reversed?
            stride_ptr = builder.gep(strides, [_const_int(i)])
            stride = builder.load(stride_ptr)
            index = caster.cast(index, stride.type)
            offset = caster.cast(offset, stride.type)
            offset = builder.add(offset, builder.mul(index, stride))

        data_ty = self.type.to_llvm(context)
        data_ptr_ty = llvm.core.Type.pointer(data_ty)

        dptr_plus_offset = builder.gep(dptr, [offset])

        ptr = builder.bitcast(dptr_plus_offset, data_ptr_ty)
        return ptr


class ArrayAttributeNode(Node):
    is_read_only = True

    _fields = ['array']

    def __init__(self, attribute_name, array):
        self.array = array
        self.attr_name = attribute_name

        self.array_type = array.variable.type
        if attribute_name == 'ndim':
            type = minitypes.int_
        elif attribute_name in ('shape', 'strides'):
            type = numba_types.SizedPointerType(numba_types.intp,
                                                size=self.array_type.ndim)
        elif attribute_name == 'data':
            type = self.array_type.dtype.pointer()
        else:
            raise error._UnknownAttribute(attribute_name)

        self.type = type

class ShapeAttributeNode(ArrayAttributeNode):
    # NOTE: better do this at code generation time, and not depend on
    #       variable.lvalue
    _fields = ['array']

    def __init__(self, array):
        super(ShapeAttributeNode, self).__init__('shape', array)
        self.array = array
        self.element_type = numba_types.intp
        self.type = minitypes.CArrayType(self.element_type,
                                         array.variable.type.ndim)

class ArrayNewNode(Node):
    """
    Allocate a new array given the attributes.
    """

    _fields = ['data', 'shape', 'strides', 'base']

    def __init__(self, type, data, shape, strides, base=None, **kwargs):
        super(ArrayNewNode, self).__init__(**kwargs)
        self.type = type
        self.data = data
        self.shape = shape
        self.strides = strides
        self.base = base

class ArrayNewEmptyNode(Node):
    """
    Allocate a new array with data.
    """

    _fields = ['shape']

    def __init__(self, type, shape, is_fortran=False, **kwargs):
        super(ArrayNewEmptyNode, self).__init__(**kwargs)
        self.type = type
        self.shape = shape
        self.is_fortran = is_fortran


class ExtTypeAttribute(Node):

    _fields = ['value']

    def __init__(self, value, attr, ctx, ext_type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.variable = ext_type.symtab[attr]
        self.type = self.variable.type
        self.ctx = ctx
        self.ext_type = ext_type

class NewExtObjectNode(Node):
    """
    Instantiate an extension type. Currently unused.
    """

    _fields = ['args']

    def __init__(self, ext_type, args, **kwargs):
        super(NewExtObjectNode, self).__init__(**kwargs)
        self.ext_type = ext_type
        self.args = args


class StructAttribute(ExtTypeAttribute):

    _fields = ['value']

    def __init__(self, value, attr, ctx, struct_type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.ctx = ctx
        self.struct_type = struct_type

        self.attr_type = struct_type.fielddict[attr]
        self.field_idx = struct_type.fields.index((attr, self.attr_type))

        self.type = self.attr_type

class ComplexNode(Node):
    _fields = ['real', 'imag']
    type = complex128
    variable = Variable(type)

class WithPythonNode(Node):
    "with python: ..."

    _fields = ['body']

class WithNoPythonNode(WithPythonNode):
    "with nopython: ..."

class ExtensionMethod(Node):

    _fields = ['value']
    call_node = None

    def __init__(self, object, attr, **kwargs):
        super(ExtensionMethod, self).__init__(**kwargs)
        ext_type = object.variable.type
        assert ext_type.is_extension
        self.value = object
        self.attr = attr

        method_type, self.vtab_index = ext_type.methoddict[attr]
        self.type = minitypes.FunctionType(return_type=method_type.return_type,
                                           args=method_type.args,
                                           is_bound_method=True)

#class ExtensionMethodCall(Node):
#    """
#    Low level call that has resolved the virtual method.
#    """
#
#    _fields = ['vmethod', 'args']
#
#    def __init__(self, vmethod, self_obj, args, signature, **kwargs):
#        super(ExtensionMethodCall, self).__init__(**kwargs)
#        self.vmethod = vmethod
#        self.args = args
#        self.signature = signature
#        self.type = signature


class ClosureNode(Node):
    """
    Inner functions or closures.

    When coerced to an object, a wrapper PyMethodDef gets created, and at
    call time a function is dynamically created with the closure scope.
    """

    _fields = []

    def __init__(self, func_def, closure_type, outer_py_func, **kwargs):
        super(ClosureNode, self).__init__(**kwargs)
        self.func_def = func_def
        self.type = closure_type
        self.outer_py_func = outer_py_func

        # self.make_pyfunc()

        self.lfunc = None
        self.wrapper_func = None
        self.wrapper_lfunc = None

        # ast and symtab after type inference
        self.type_inferred_ast = None
        self.symtab = None

        # The Python extension type that must be instantiated to hold cellvars
        # self.scope_type = None
        self.ext_type = None
        self.need_closure_scope = False

        # variables we need to put in a closure scope for our inner functions
        # This is set on the FunctionDef node
        # self.cellvars = None

    def make_pyfunc(self):
        d = self.outer_py_func.func_globals
#        argnames = tuple(arg.id for arg in self.func_def.args.args)
#        dummy_func_string = """
#def __numba_closure_func(%s):
#    pass
#        """ % ", ".join(argnames)
#        exec dummy_func_string in d, d

        name = self.func_def.name
        self.func_def.name = '__numba_closure_func'
        ast_mod = ast.Module(body=[self.func_def])
        numba.functions.fix_ast_lineno(ast_mod)
        c = compile(ast_mod, '<string>', 'exec')
        exec c in d, d
        self.func_def.name = name

        self.py_func = d['__numba_closure_func']
        self.py_func.live_objects = []
        self.py_func.__module__ = self.outer_py_func.__module__
        self.py_func.__name__ = name

class InstantiateClosureScope(Node):

    _fields = ['outer_scope']

    def __init__(self, func_def, scope_ext_type, scope_type, outer_scope, **kwargs):
        super(InstantiateClosureScope, self).__init__(**kwargs)
        self.func_def = func_def
        self.scope_type = scope_type
        self.ext_type = scope_ext_type
        self.outer_scope = outer_scope
        self.type = scope_type

class ClosureScopeLoadNode(Node):
    "Load the closure scope for the function or NULL"

    type = object_

class ClosureCallNode(NativeCallNode):
    """
    Call to closure or inner function.
    """

    def __init__(self, closure_type, call_node, **kwargs):
        self.call_node = call_node
        args, keywords = call_node.args, call_node.keywords
        args = args + self._resolve_keywords(closure_type, args, keywords)
        super(ClosureCallNode, self).__init__(closure_type.signature, args,
                                              llvm_func=None, **kwargs)
        self.closure_type = closure_type

    def _resolve_keywords(self, closure_type, args, keywords):
        func_def = closure_type.closure.func_def
        argnames = [name.id for name in func_def.args.args]

        expected = len(argnames) - len(args)
        if len(keywords) != expected:
            raise error.NumbaError(
                    self.call_node,
                    "Expected %d arguments, got %d" % (len(argnames),
                                                       len(args) + len(keywords)))

        argpositions = dict(zip(argnames, range(len(argnames))))
        positional = [None] * (len(argnames) - len(args))

        for keyword in keywords:
            argname = keyword.arg
            pos = argpositions.get(argname, None)
            if pos is None:
                raise error.NumbaError(
                        keyword, "Not a valid keyword argument name: %s" % argname)
            elif pos < len(args):
                raise error.NumbaError(
                        keyword, "Got multiple values for positional "
                                 "argument %r" % argname)
            else:
                positional[pos] = keyword.value

        return positional

class FunctionWrapperNode(Node):
    """
    This code is a wrapper function callable from Python using PyCFunction:

        PyObject *(*)(PyObject *self, PyObject *args)

    It unpacks the tuple to native types, calls the wrapped function, and
    coerces the return type back to an object.
    """

    _fields = ['body', 'return_result']

    def __init__(self, wrapped_function, signature, orig_py_func, fake_pyfunc):
        self.wrapped_function = wrapped_function
        self.signature = signature
        self.orig_py_func = orig_py_func
        self.fake_pyfunc = fake_pyfunc

def pointer_add(pointer, offset):
    assert pointer.type == char.pointer()
    left = ptrtoint(pointer)
    result = ast.BinOp(left, ast.Add(), offset)
    result.type = left.type
    result.variable = Variable(result.type)
    return CoercionNode(result, char.pointer())

class DereferenceNode(Node):
    """
    Dereference a pointer
    """

    _fields = ['pointer']

    def __init__(self, pointer, **kwargs):
        super(DereferenceNode, self).__init__(**kwargs)
        self.pointer = pointer
        self.type = pointer.type.base_type


class PointerFromObject(Node):
    """
    Bitcast objects to void *
    """

    _fields = ['node']
    type = void.pointer()
    variable = Variable(type)

    def __init__(self, node, **kwargs):
        super(PointerFromObject, self).__init__(**kwargs)
        self.node = node


#
### Nodes for NumPy calls
#

shape_type = npy_intp.pointer()
void_p = void.pointer()

class MultiArrayAPINode(NativeCallNode):

    def __init__(self, name, signature, args):
        super(MultiArrayAPINode, self).__init__(signature, args,
                                                llvm_func=None)
        self.func_name = name

def PyArray_NewFromDescr(args):
    """
    Low-level specialized equivalent of ArrayNewNode
    """
    signature = object_(
        object_,    # subtype
        object_,    # descr
        int_,       # ndim
        shape_type, # shape
        shape_type, # strides
        void_p,     # data
        int_,       # flags
        object_,    # obj
    )

    return MultiArrayAPINode('PyArray_NewFromDescr', signature, args)

def PyArray_SetBaseObject(args):
    signature = int_(object_, object_)
    return MultiArrayAPINode('PyArray_SetBaseObject', signature, args)

def PyArray_UpdateFlags(args):
    return MultiArrayAPINode('PyArray_UpdateFlags', void(object_, int_), args)

def PyArray_Empty(args, name='PyArray_Empty'):
    nd, shape, dtype, fortran = args
    return_type = minitypes.ArrayType(dtype, nd)
    signature = return_type(
                int_,                   # nd
                npy_intp.pointer(),     # shape
                object_,                # dtype
                int_)                   # fortran
    return MultiArrayAPINode(name, signature, args)

def PyArray_Zeros(args):
    return PyArray_Empty(args, name='PyArray_Zeros')
