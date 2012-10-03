import ast
import copy
import opcode
import types
import __builtin__ as builtins

import numba
from numba import *
from numba import error
from .minivect import minierror, minitypes
from . import translate, utils, _numba_types as _types
from .symtab import Variable
from . import visitors, nodes, error
from numba import stdio_util

logger = logging.getLogger(__name__)


class TransformForIterable(visitors.NumbaTransformer):
    def __init__(self, context, func, ast, symtab):
        super(TransformForIterable, self).__init__(context, func, ast)
        self.symtab = symtab

    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            self.error(node.target,
                       "Only assignment to target names is supported.")

        # NOTE: this would be easier to do during type inference, since you
        #       can rewrite the AST and run the type inferer on the rewrite.
        #       This way, you won't have to create variables and types manually.
        #       This will also take care of proper coercions.
        if node.iter.type.is_range:
            # make sure to analyse children, in case of nested loops
            self.generic_visit(node)
            node.index = ast.Name(id=node.target.id, ctx=ast.Load())
            return node
        elif node.iter.type.is_array and node.iter.type.ndim == 1:
            # Convert 1D array iteration to for-range and indexing
            logger.debug(ast.dump(node))

            orig_target = node.target
            orig_iter = node.iter

            # replace node.target with a temporary
            target_name = orig_target.id + '.idx'
            target_temp = nodes.TempNode(minitypes.Py_ssize_t)
            node.target = target_temp.store()

            # replace node.iter
            call_func = ast.Name(id='range', ctx=ast.Load())
            call_func.type = _types.RangeType()
            shape_index = ast.Index(nodes.ConstNode(0, _types.Py_ssize_t))
            shape_index.type = _types.npy_intp
            stop = ast.Subscript(value=nodes.ShapeAttributeNode(orig_iter),
                                 slice=shape_index,
                                 ctx=ast.Load())
            stop.type = _types.intp
            call_args = [nodes.ConstNode(0, _types.Py_ssize_t),
                         nodes.CoercionNode(stop, _types.Py_ssize_t),
                         nodes.ConstNode(1, _types.Py_ssize_t),]

            node.iter = ast.Call(func=call_func, args=call_args)
            node.iter.type = call_func.type

            node.index = target_temp.load()
            # add assignment to new target variable at the start of the body
            index = ast.Index(value=node.index)
            index.type = target_temp.type
            subscript = ast.Subscript(value=orig_iter,
                                      slice=index, ctx=ast.Load())
            subscript.type = orig_iter.variable.type.dtype
            coercion = nodes.CoercionNode(subscript, orig_target.type)
            assign = ast.Assign(targets=[orig_target], value=subscript)

            node.body = [assign] + node.body

            return node
        else:
            raise error.NumbaError("Unsupported for loop pattern")

class LateSpecializer(visitors.NumbaTransformer):

    def __init__(self, context, func, ast, func_signature):
        super(LateSpecializer, self).__init__(context, func, ast)
        self.func_signature = func_signature

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

        ret_type = self.func_signature.return_type
        if ret_type.is_object or ret_type.is_array:
            value = nodes.NULL_obj
        elif ret_type.is_void:
            value = None
        elif ret_type.is_float:
            value = nodes.ConstNode(float('nan'), type=ret_type)
        elif ret_type.is_int or ret_type.is_complex:
            value = nodes.ConstNode(0xbadbadbad, type=ret_type)
        else:
            value = None

        if value is not None:
            value = nodes.CoercionNode(value, dst_type=ret_type)

        node.error_return = ast.Return(value=value)
        return node

    def _print(self, value, dest=None):
        stdin, stdout, stderr = stdio_util.get_stdio_streams()
        stdout = stdio_util.get_stream_as_node(stdout)

        signature, lfunc = self.function_cache.function_by_name(
                                                'PyObject_CallMethod')
        if dest is None:
            dest = nodes.ObjectInjectNode(sys.stdout)

        value = self.function_cache.call("PyObject_Str", value)
        args = [dest, nodes.ConstNode("write"), nodes.ConstNode("O"), value]
        return nodes.NativeCallNode(signature, args, lfunc)

    def visit_Print(self, node):
        result = []
        for value in node.values:
            value = nodes.CoercionNode(value, object_, name="print_arg")
            result.append(self._print(value, node.dest))

        if node.nl:
            result.append(self._print(nodes.ObjectInjectNode("\n"), node.dest))

        return self.visitlist(result)

    def visit_Tuple(self, node):
        sig, lfunc = self.function_cache.function_by_name('PyTuple_Pack')
        objs = self.visitlist(nodes.CoercionNode.coerce(node.elts, object_))
        n = nodes.ConstNode(len(node.elts), minitypes.Py_ssize_t)
        args = [n] + objs
        new_node = nodes.NativeCallNode(sig, args, lfunc, name='tuple')
        new_node.type = _types.TupleType(size=len(node.elts))
        return nodes.ObjectTempNode(new_node)

    def visit_Dict(self, node):
        self.generic_visit(node)
        return nodes.ObjectTempNode(node)

    def visit_NativeCallNode(self, node):
        self.generic_visit(node)
        if node.signature.return_type.is_object:
            node = nodes.ObjectTempNode(node)
        return node

    def visit_ObjectCallNode(self, node):
        # self.generic_visit(node)
        assert node.function
        node.function = self.visit(node.function)
        node.args_tuple = self.visit(node.args_tuple)
        node.kwargs_dict = self.visit(node.kwargs_dict)
        return nodes.ObjectTempNode(node)

    def visit_CoercionNode(self, node, visitchildren=True):
        if visitchildren:
            self.generic_visit(node)
        elif not isinstance(node, nodes.CoercionNode):
            # CoercionNode.__new__ returns the node to be coerced if it doesn't
            # need coercion
            return node

        node_type = node.node.type
        if node.dst_type.is_object and not node_type.is_object:
            return nodes.ObjectTempNode(nodes.CoerceToObject(
                    node.node, node.dst_type, name=node.name))
        elif node_type.is_object and not node.dst_type.is_object:
            # Create a tuple for PyArg_ParseTuple
            # TODO: efficient conversions
            tup = ast.Tuple(elts=[node.node], ctx=ast.Load())
            tup = self.visit(tup)
            return nodes.CoerceToNative(tup, node.dst_type, name=node.name)
        return node

    def visit_Subscript(self, node):
        if isinstance(node.value, nodes.ArrayAttributeNode):
            if node.value.is_read_only and isinstance(node.ctx, ast.Store):
                raise error.NumbaError("Attempt to load read-only attribute")

        # logging.debug(ast.dump(node))
        self.generic_visit(node)

        node_type = node.value.type
        if node_type.is_object or (node_type.is_array and
                                   node.slice.type.is_object):
            # Array or object slicing
            if isinstance(node.ctx, ast.Load):
                result = self.function_cache.call('PyObject_GetItem',
                                                  node.value, node.slice)
                # print ast.dump(result)
                node = nodes.CoercionNode(result, dst_type=node.type)
                node = self.visit_CoercionNode(node, visitchildren=False)
            else:
                # This is handled in visit_Assign
                pass
        elif (node.value.type.is_array and not node.type.is_array and
                  node.slice.type.is_int):
            # Array index with integer indices
            node = nodes.DataPointerNode(node.value, node.slice, node.ctx)

        return node

    def visit_ExtSlice(self, node):
        if node.type.is_object:
            return self.visit(ast.Tuple(elts=node.dims, ctx=ast.Load()))
        else:
            self.generic_visit(node)
            return node

    def visit_Assign(self, node):
        target = node.targets[0]
        if (len(node.targets) == 1 and
                isinstance(target, ast.Subscript) and
                (target.type.is_array or target.type.is_object)):
            # Slice assignment / index assignment w/ objects
            # TODO: discount array indexing with dtype object
            target = self.visit(target)
            obj = target.value
            key = target.slice
            value = self.visit(node.value)
            call = self.function_cache.call('PyObject_SetItem',
                                            obj, key, value)
            return call

        self.generic_visit(node)
        return node

    def visit_Slice(self, node):
        """
        Rewrite slice objects. Do this late in the pipeline so that other
        code can still recognize the code structure.
        """
        slice_values = [node.lower, node.upper, node.step]

        if all(isinstance(node, nodes.ConstNode) for node in slice_values):
            get_const = lambda node: None if node is None else node.pyval
            value = slice(get_const(node.lower), get_const(node.upper),
                          get_const(node.step))
            return self.visit(nodes.ObjectInjectNode(value))

        bounds = []
        for node in slice_values:
            if node is None:
                bounds.append(nodes.NULL_obj)
            else:
                bounds.append(node)

        new_slice = self.function_cache.call('PySlice_New', *bounds,
                                             temp_name='slice')
        return self.visit(new_slice)
        # return nodes.ObjectTempNode(new_slice)

    def visit_Attribute(self, node):
        if node.type.is_numpy_attribute:
            return nodes.ObjectInjectNode(node.type.value)

        new_node = self.function_cache.call(
                            'PyObject_GetAttrString', node.value,
                            nodes.ConstNode(node.attr))
        self.generic_visit(new_node)
        return new_node

    def visit_Name(self, node):
        if node.type.is_builtin:
            obj = getattr(builtins, node.name)
            return nodes.ObjectInjectNode(obj, node.type)

        return node

    def visit_Return(self, node):
        return_type = self.func_signature.return_type
        if node.value is not None:
            node.value = self.visit(nodes.CoercionNode(node.value, return_type))
        return node