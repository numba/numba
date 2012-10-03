import ast
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


def _const_int(X):
    return llvm.core.Constant.int(llvm.core.Type.int(), X)

class Node(ast.AST):
    """
    Superclass for Numba AST nodes
    """
    _fields = []

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

class CoercionNode(Node):
    """
    Coerce a node to a different type
    """

    _fields = ['node']

    def __new__(cls, node, dst_type, name=''):
        type = getattr(node, 'type', None) or node.variable.type
        if type == dst_type:
            return node
        return super(CoercionNode, cls).__new__(cls, node, dst_type, name=name)

    def __init__(self, node, dst_type, name=''):
        if node is self:
            # We are trying to coerce a CoercionNode which already has the
            # right type, so __new__ returns a CoercionNode, which then results
            # in __init__ being called
            return

        self.node = node
        self.dst_type = dst_type
        self.variable = Variable(dst_type)
        self.type = dst_type
        self.name = name

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
            base_ltype = self.to_llvm(type.base_type)
            lvalue = llvm.core.Constant.struct([(base_ltype, constant.real),
                                                (base_ltype, constant.imag)])
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


_NULL = object()
NULL_obj = ConstNode(_NULL, object_)

def const(obj, type):
    if type.is_object:
        node = ObjectInjectNode(obj, type)
    else:
        node = ConstNode(obj)

    return node


class FunctionCallNode(Node):
    def __init__(self, signature, args, name=''):
        self.signature = signature
        self.variable = Variable(signature.return_type)
        self.name = name
        self.original_args = args


class NativeCallNode(FunctionCallNode):
    _fields = ['args']

    def __init__(self, signature, args, llvm_func, py_func=None, **kw):
        super(NativeCallNode, self).__init__(signature, args, **kw)
        self.llvm_func = llvm_func
        self.py_func = py_func
        self.coerce_args()
        self.type = signature.return_type

    def coerce_args(self):
        self.args = list(self.original_args)
        for i, dst_type in enumerate(self.signature.args):
            self.args[i] = CoercionNode(self.args[i], dst_type,
                                        name='func_%s_arg%d' % (self.name, i))

class LLVMIntrinsicNode(NativeCallNode):
    "Call an llvm intrinsic function"

class MathCallNode(NativeCallNode):
    "Call a math function"

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


class ObjectInjectNode(Node):
    """
    Refer to a Python object in the llvm code.
    """

    def __init__(self, object, type=None, **kwargs):
        super(ObjectInjectNode, self).__init__(**kwargs)
        self.object = object
        self.type = type or object_
        self.variable = Variable(self.type)


class ObjectTempNode(Node):
    """
    Coerce a node to a temporary which is reference counted.
    """

    _fields = ['node']

    def __init__(self, node):
        assert not isinstance(node, ObjectTempNode)
        self.node = node
        self.llvm_temp = None
        self.type = getattr(node, 'type', node.variable.type)
        self.variable = Variable(self.type)


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

class TempStoreNode(Node):
    _fields = ['temp']

class DataPointerNode(Node):

    _fields = ['node', 'index']

    def __init__(self, node, slice, ctx):
        self.node = node
        self.slice = slice
        self.variable = Variable(node.type)
        self.type = node.type
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

        data_ty = self.variable.type.dtype.to_llvm(context)
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
            type = minitypes.CArrayType(numba_types.intp, array_type.ndim)
        elif attribute_name == 'data':
            type = array_type.dtype.pointer()
        else:
            raise error._UnknownAttribute(attribute_name)

        self.type = type
        self.variable = Variable(type)

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
        self.variable = Variable(self.type)

