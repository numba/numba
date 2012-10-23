import ast
import ctypes
import collections

import numba
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

def const(obj, type):
    if type.is_object:
        node = ObjectInjectNode(obj, type)
    else:
        node = ConstNode(obj)

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

class CoercionNode(Node):
    """
    Coerce a node to a different type
    """

    _fields = ['node']

    def __new__(cls, node, dst_type, name=''):
        type = getattr(node, 'type', None) or node.variable.type
        #if type == dst_type:
        #    return node

        if isinstance(node, ConstNode) and dst_type.is_numeric:
            node.cast(dst_type)
            return node

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


class DeferredCoercionNode(CoercionNode):
    """
    Coerce to the type of the given variable. The type of the variable may
    change in the meantime (e.g. may be promoted or demoted).
    """

    _fields = ['node']

    def __init__(self, node, variable):
        if node is self:
            return
        self.node = node
        self.variable = variable


class ConstNode(Node):
    """
    Wrap a constant.
    """

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

        if type.is_float:
            lvalue = llvm.core.Constant.real(ltype, constant)
        elif type.is_int:
            lvalue = llvm.core.Constant.int(ltype, constant)
        elif type.is_complex:
            real = ConstNode(constant.real, type.base_type)
            imag = ConstNode(constant.imag, type.base_type)
            lvalue = llvm.core.Constant.struct([real.value(translator),
                                                imag.value(translator)])
        elif type.is_pointer:
            addr_int = translator.visit(ConstNode(self.pyval, type=Py_ssize_t))
            lvalue = translator.builder.inttoptr(addr_int, ltype)
        elif type.is_object:
            if self.pyval is _NULL:
                lvalue = llvm.core.Constant.null(type.to_llvm(context))
            else:
                raise NotImplementedError("Use ObjectInjectNode")
        elif type.is_c_string:
            lvalue = translate._LLVMModuleUtils.get_string_constant(
                                            translator.mod, constant)
            type_char_p = numba_types.c_string_type.to_llvm(translator.context)
            lvalue = translator.builder.bitcast(lvalue, type_char_p)
        elif type.is_function:
            # TODO:
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

class FunctionCallNode(Node):
    def __init__(self, signature, args, name=''):
        self.signature = signature
        self.type = signature.return_type
        self.name = name
        self.original_args = args

class NativeCallNode(FunctionCallNode):
    _fields = ['args']

    def __init__(self, signature, args, llvm_func, py_func=None,
                 skip_self=False, **kw):
        super(NativeCallNode, self).__init__(signature, args, **kw)
        self.llvm_func = llvm_func
        self.py_func = py_func
        self.skip_self = skip_self
        self.type = signature.return_type
        self.coerce_args()

    def coerce_args(self):
        self.args = list(self.original_args)
        for i, dst_type in enumerate(self.signature.args[self.skip_self:]):
            self.args[i] = CoercionNode(self.args[i], dst_type,
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

class LLVMIntrinsicNode(NativeCallNode):
    "Call an llvm intrinsic function"

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


class ObjectInjectNode(Node):
    """
    Refer to a Python object in the llvm code.
    """

    _attributes = ['object', 'type']

    def __init__(self, object, type=None, **kwargs):
        super(ObjectInjectNode, self).__init__(**kwargs)
        self.object = object
        self.type = type or object_


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
    re-used without
    """

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(CloneableNode, self).__init__(**kwargs)
        self.node = node
        self.clone_nodes = []
        self.type = node.type

class CloneNode(Node):

    _fields = ['node']

    def __init__(self, node, **kwargs):
        super(CloneNode, self).__init__(**kwargs)

        assert isinstance(node, CloneableNode)
        self.node = node
        self.type = node.type
        node.clone_nodes.append(self)

        self.llvm_value = None

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

        array_type = array.variable.type
        if attribute_name == 'ndim':
            type = minitypes.int_
        elif attribute_name in ('shape', 'strides'):
            type = numba_types.SizedPointerType(numba_types.intp,
                                                size=array_type.ndim)
        elif attribute_name == 'data':
            type = array_type.dtype.pointer()
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


class ExtTypeAttribute(Node):

    _fields = ['value']

    def __init__(self, value, attr, ctx, ext_type, **kwargs):
        super(ExtTypeAttribute, self).__init__(**kwargs)
        self.value = value
        self.attr = attr
        self.variable = ext_type.symtab[attr]
        self.ctx = ctx
        self.ext_type = ext_type


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
    left = CoercionNode(pointer, Py_ssize_t)
    result = ast.BinOp(left, ast.Add(), offset)
    result.type = Py_ssize_t
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
