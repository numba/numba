import ast
import opcode
import types
import __builtin__ as builtins

from .minivect import minierror, minitypes
from . import translate, utils, _numba_types as _types
from .symtab import Variable
from . import visitors, nodes

import numpy

def _infer_types(context, func, ast, func_signature):
    """
    Run type inference on the given ast.
    """
    type_inferer = TypeInferer(context, func, ast,
                               func_signature=func_signature)
    type_inferer.infer_types()
    TypeSettingVisitor(context, func, ast).visit(ast)
    return type_inferer.func_signature, type_inferer.symtab

class TypeInferer(visitors.NumbaTransformer):
    """
    Type inference. Initialize with a minivect context, a Python ast,
    and a function type with a given or absent, return type.

    Infers and checks types and inserts type coercion nodes.
    """

    def __init__(self, context, func, ast, func_signature):
        super(TypeInferer, self).__init__(context, func, ast)
        # Name -> Variable
        self.symtab = {}

        self.func_signature = func_signature
        self.given_return_type = func_signature.return_type
        self.return_variables = []
        self.return_type = None

    def infer_types(self):
        """
        Infer types for the function.
        """
        self.init_locals()

        self.return_variable = Variable(None)
        self.ast = self.visit(self.ast)

        self.return_type = self.return_variable.type
        ret_type = self.func_signature.return_type
        if ret_type and ret_type != self.return_type:
            self.assert_assignable(ret_type, self.return_type)
            self.return_type = self.promote_types(ret_type, self.return_type)

        self.func_signature = minitypes.FunctionType(
                return_type=self.return_type, args=self.func_signature.args)

    def init_global(self, global_name):
        globals = self.func.__globals__
        # Determine the type of the global, i.e. a builtin, global
        # or (numpy) module
        if (global_name not in globals and
                getattr(builtins, global_name, None)):
            type = _types.BuiltinType(name=global_name)
        else:
            # FIXME: analyse the bytecode of the entire module, to determine
            # overriding of builtins
            if isinstance(globals.get(global_name), types.ModuleType):
                type = _types.ModuleType()
                type.is_numpy_module = globals[global_name] is numpy
                if type.is_numpy_module:
                    type.module = numpy
            else:
                type = _types.GlobalType(name=global_name)

        variable = Variable(type, name=global_name)
        self.symtab[global_name] = variable
        return variable

    def init_locals(self):
        arg_types = self.func_signature.args
        for i, arg_type in enumerate(arg_types):
            varname = self.local_names[i]
            self.symtab[varname] = Variable(arg_type, is_local=True, name=varname)

        for varname in self.varnames[len(arg_types):]:
            self.symtab[varname] = Variable(None, is_local=True, name=varname)

        self.symtab['None'] = Variable(minitypes.object_, is_constant=True,
                                       constant_value=None)

    def promote_types(self, t1, t2):
        return self.context.promote_types(t1, t2)

    def promote(self, v1, v2):
        return self.promote_types(v1.type, v2.type)

    def assert_assignable(self, dst_type, src_type):
        self.promote_types(dst_type, src_type)

    def type_from_pyval(self, pyval):
        return self.context.typemapper.from_python(pyval)

    def visit(self, node):
        if node is Ellipsis:
            node = ast.Ellipsis()
        return super(TypeInferer, self).visit(node)

    def visit_AugAssign(self, node):
        "Inplace assignment"
        target = node.target
        if isinstance(target, ast.Name):
            target = ast.copy_location(ast.Name(target.id, ast.Store()), target)
            rhs_target = ast.copy_location(ast.Name(target.id, ast.Load()), target)
        else:
            raise NotImplementedError("Inplace assignment on non-variable target not supported")

        assignment = ast.Assign([target], ast.BinOp(rhs_target, node.op, node.value))
        return self.visit(assignment)

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise NotImplementedError('Mutliple targets in assignment.')

        node.value = self.visit(node.value)
        target = self.visit(node.targets[0])
        node.targets[0] = self.assign(target.variable, node.value.variable, target)
        return node

    def assign(self, lhs_var, rhs_var, rhs_node):
        if lhs_var.type is None:
            lhs_var.type = rhs_var.type
        elif lhs_var.type != rhs_var.type:
            self.assert_assignable(lhs_var.type, rhs_var.type)
            return nodes.CoercionNode(rhs_node, lhs_var.type)

        return rhs_node

    def _get_iterator_type(self, iterator_type):
        "Get the type of an iterator Variable"
        if iterator_type.is_iterator:
            base_type = iterator_type.base_type
        elif iterator_type.is_array:
            base_type = iterator_type.dtype
        elif iterator_type.is_range:
            base_type = _types.Py_ssize_t
        else:
            raise NotImplementedError("Unknown type: %s" % (iterator_type,))

        return base_type

    def visit_For(self, node):
        if node.orelse:
            raise NotImplementedError('Else in for-loop is not implemented.')

        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        base_type = self._get_iterator_type(node.iter.variable.type)
        node.target = self.assign(node.target.variable, Variable(base_type),
                                  node.target)
        self.visitlist(node.body)
        return node

    def visit_While(self, node):
        if node.orelse:
            raise NotImplementedError('Else in for-loop is not implemented.')
        node.test = CoercionNode(self.visit(node.test), minitypes.bool_)
        node.body = self.visit(node.body)
        return node

    def visit_Name(self, node):
        variable = self.symtab.get(node.id)
        if variable:
            # local variable
            if (variable.type is None and variable.is_local and
                    isinstance(node.ctx, ast.Load)):
                raise UnboundLocalError(variable.name)
        else:
            variable = self.init_global(node.id)

        node.variable = variable
        return node

    def visit_BoolOp(self, node):
        "and/or expression"
        if len(node.values) != 2:
            raise AssertionError

        node.values = self.visit(node.values)
        node.values[:] = CoercionNode.coerce(node.values, minitypes.bool_)
        node.variable = Variable(minitypes.bool_)
        return node

    def visit_BinOp(self, node):
        # TODO: handle floor devision
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        promotion_type = self.promote(node.left.variable,
                                      node.right.variable)
        node.left, node.right = nodes.CoercionNode.coerce(
                            [node.left, node.right], promotion_type)
        node.variable = Variable(promotion_type)
        return node

    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            node.operand = CoercionNode(node.operand, minitypes.bool_)
            return self.setvar(node, Variable(minitypes.bool_))

        node.variable = Variable(node.variable.type)
        return node

    def visit_Compare(self, node):
        if len(node.ops) != 1:
            raise NotImplementedError('Multiple operators not supported')

        if len(node.comparators)!=1:
            raise NotImplementedError('Multiple comparators not supported')

        self.generic_visit(node)
        node.variable = Variable(minitypes.bool_)
        return node

    def _get_index_type(self, type, index_type):
        if type.is_pointer:
            assert index_type.is_int_like
            return type.base_type
        elif type.is_object:
            return minitypes.object_
        else:
            raise NotImplementedError

    def _is_newaxis(self, node):
        v = node.variable
        return ((v.type.is_object and v.constant_value is None) or
                v.type.is_newaxis)

    def _unellipsify(self, node, slices, subscript_node):
        """
        Given an array node `node`, process all AST slices and create the
        final type:

            - process newaxes (None or numpy.newaxis)
            - replace Ellipsis with a bunch of ast.Slice objects
            - process integer indices
            - append any missing slices in trailing dimensions
        """
        type = node.variable.type

        if not type.is_array:
            assert type.is_object
            return minitypes.object_, node

        result = []
        seen_ellipsis = False

        # Filter out newaxes
        newaxes = [newaxis for newaxis in slices if self._is_newaxis(newaxis)]
        n_indices = len(slices) - len(newaxes)

        full_slice = ast.Slice(lower=None, upper=None, step=None)
        ast.copy_location(full_slice, slices[0])

        # process ellipses and count integer indices
        indices_seen = 0
        for slice_node in slices[::-1]:
            slice_type = slice_node.variable.type
            if slice_type.is_ellipsis:
                if seen_ellipsis:
                    result.append(full_slice)
                else:
                    nslices = type.ndim - n_indices + 1
                    result.extend([full_slice] * nslices)
                    seen_ellipsis = True
            elif (slice_type.is_slice or slice_type.is_int or
                      self._is_newaxis(slice_node)):
                indices_seen += slice_type.is_int
                result.append(slice_node)
            else:
                # TODO: Coerce all object operands to integer indices?
                # TODO: (This will break indexing with the Ellipsis object or
                # TODO:  with slice objects that we couldn't infer)
                return minitypes.object_, nodes.CoercionNode(node,
                                                             minitypes.object_)

        # append any missing slices (e.g. a2d[:]
        result_length = len(result) - len(newaxes)
        if result_length < type.ndim:
            nslices = type.ndim - result_length
            result.extend([full_slice] * nslices)

        subscript_node.slice = ast.ExtSlice(result)
        ast.copy_location(subscript_node.slice, slices[0])

        # create the final array type and set it in value.variable
        result_dtype = node.variable.type.dtype
        result_ndim = node.variable.type.ndim + len(newaxes) - indices_seen
        if result_ndim > 0:
            result_type = result_dtype[(slice(None),) * result_ndim]
        elif result_ndim == 0:
            result_type = result_dtype
        else:
            result_type = minitypes.object_

        return result_type, node

    def visit_Subscript(self, node):
        node.value = self.visit(node.value)
        node.slice = self.visit(node.slice)

        if not node.value.variable.type.is_array:
            result_type = self._get_index_type(node.value.variable.type,
                                               node.slice.variable.type)
        else:
            slice_variable = node.slice.variable
            slice_type = slice_variable.type
            if (slice_type.is_tuple and
                    isinstance(node.slice, ast.Index)):
                node.slice = node.slice.value

            slices = None
            if (isinstance(node.slice, ast.Index) or
                    slice_type.is_ellipsis or slice_type.is_slice):
                slices = [node.slice]
            elif isinstance(node.slice, ast.ExtSlice):
                slices = list(node.slice.dims)
            elif isinstance(node.slice, ast.Tuple):
                slices = list(node.slice.elts)

            if slices is None:
                result_type = minitypes.object_
            else:
                result_type, node.value = self._unellipsify(node.value, slices, node)

        node.variable = Variable(result_type)
        return node

    def visit_Index(self, node):
        "Normal index"
        node.value = self.visit(node.value)
        variable = node.value.variable
        type = variable.type
        if (type.is_object and variable.is_constant and
                variable.constant_value is None):
            type = _types.NewAxisType()

        node.variable = Variable(type)
        return node

    def visit_Ellipsis(self, node):

        return nodes.ConstNode(Ellipsis, _types.EllipsisType())

    def visit_Slice(self, node):
        self.generic_visit(node)
        type = _types.SliceType()

        values = [node.lower, node.upper, node.step]
        constants = []
        for value in values:
            if value is None:
                constants.append(None)
            elif constant.variable.is_constant:
                constants.append(value.variable.constant_value)
            else:
                break
        else:
            return nodes.ConstNode(slice(*constants), type)

        node.variable = Variable(type)
        return node

    def visit_ExtSlice(self, node):
        self.generic_visit(node)
        node.variable = Variable(minitypes.object_)
        return node

    def visit_Num(self, node):
        return nodes.ConstNode(node.n)

    def visit_Str(self, node):
        return nodes.ConstNode(node.s)

    def _get_constant_list(self, node):
        if not isinstance(node.ctx, ast.Load):
            return None

        items = []
        constant_value = None
        for item_node in node.elts:
            if item_node.variable.is_constant:
                items.append(item_node.variable.constant_value)
            else:
                return None

        return items

    def visit_Tuple(self, node):
        self.visitlist(node.elts)
        constant_value = self._get_constant_list(node)
        if constant_value is not None:
            constant_value = tuple(constant_value)
        type = _types.TupleType(size=len(node.elts),
                                is_constant=constant_value is not None,
                                constant_value=constant_value)
        node.variable = Variable(type)
        return node

    def visit_List(self, node):
        self.visitlist(node.elts)
        constant_value = self._get_constant_list(node)
        type = _types.ListType(size=len(node.elts),
                               is_constant=constant_result is not None,
                               constant_value=constant_value)
        node.variable = Variable(type)
        return node

    def _resolve_attribute_dtype(self, dtype):
        "Resolve the type for numpy dtype attributes"
        if dtype.is_numpy_attribute:
            numpy_attr = getattr(dtype.module, dtype.attr, None)
            if isinstance(numpy_attr, numpy.dtype):
                return _types.NumpyDtypeType(dtype=numpy_attr)
            elif issubclass(numpy_attr, numpy.generic):
                return _types.NumpyDtypeType(dtype=numpy.dtype(numpy_attr))

    def _get_dtype(self, node, args, dtype=None):
        "Get the dtype keyword argument from a call to a numpy attribute."
        for keyword in node.keywords:
            if keyword.arg == 'dtype':
                dtype = keyword.value.variable.type
                return self._resolve_attribute_dtype(dtype)
        else:
            # second argument is a dtype
            if args and args[0].variable.type.is_numpy_dtype:
                return args[0].variable.type

        return dtype

    def _resolve_empty_like(self, node):
        "Parse the result type for np.empty_like calls"
        args = node.args
        if args:
            arg = node.args[0]
            args = args[1:]
        else:
            for keyword in node.keywords:
                if keyword.arg == 'a':
                    arg = keyword.value
                    break
            else:
                return None

        type = arg.variable.type
        if type.is_array:
            dtype = self._get_dtype(node, args)
            if dtype is None:
                return type
            else:
                if not dtype.is_numpy_dtype:
                    # We have a dtype that cannot be inferred
                    # raise NotImplementedError("Uninferred dtype")
                    return None
                return minitypes.ArrayType(dtype.resolve(), type.ndim)

    def _resolve_arange(self, node):
        "Resolve a call to np.arange()"
        dtype = self._get_dtype(node, node.args, _types.NumpyDtypeType(
                                        dtype=numpy.dtype(numpy.int64)))
        if dtype is not None:
            # return a 1D array type of the given dtype
            return dtype.resolve()[:]

    def _resolve_numpy_call(self, func_type, node):
        """
        Resolve a call of some numpy attribute or sub-attribute.
        """
        numpy_func = getattr(func_type.module, func_type.attr)
        if numpy_func in (numpy.zeros_like, numpy.ones_like,
                          numpy.empty_like):
            result_type = self._resolve_empty_like(node)
        elif numpy_func is numpy.arange:
            result_type = self._resolve_arange(node)
        else:
            result_type = None

        return result_type

    def visit_Call(self, node):
        node.func = self.visit(node.func)
        self.visitlist(node.args)
        self.visitlist(node.keywords)

        func_type = node.func.variable.type
        arg_type = None
        if func_type.is_builtin and func_type.name in ('range', 'xrange'):
            result = Variable(_types.RangeType())
            arg_type = minitypes.Py_ssize_t
        elif func_type.is_builtin and func.name == 'len':
            result = Variable(_types.Py_ssize_t)
        elif func_type.is_function:
            result = Variable(func.type.return_type)
        elif func_type.is_numpy_attribute:
            result_type = self._resolve_numpy_call(func_type, node)
            if result_type is None:
                result_type = minitypes.object_
            result = Variable(result_type)
        elif func_type.is_object:
            result = Variable(minitypes.object_)
        else:
            # TODO: implement
            raise NotImplementedError(func.type)

        if arg_type is not None:
            node.args = [nodes.CoercionNode(arg, arg_type)
                            for arg in node.args]

        node.variable = result
        return node

    def _resolve_numpy_attribute(self, node, type):
        attribute = getattr(type.module, node.attr)
        if attribute is numpy.newaxis:
            result_type = _types.NewAxisType()
        else:
            result_type = _types.NumpyAttributeType(module=type.module,
                                                    attr=node.attr)
        return result_type

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        type = node.value.variable.type
        if type.is_complex:
            if node.attr in ('real', 'imag'):
                if self.is_store(node.ctx):
                    raise TypeError("Cannot assign to the %s attribute of "
                                    "complex numbers" % node.attr)
                result_type = type.base_type
            else:
                raise AttributeError("'%s' of complex type" % node.attr)
        elif type.is_numpy_module and hasattr(type.module, node.attr):
            result_type = self._resolve_numpy_attribute(node, type)
        elif type.is_object:
            result_type = type
        elif type.is_array:
            # handle shape/strides/suboffsets etc
            raise NotImplementedError
        else:
            raise NotImplementedError((node.attr, node.value, type))

        node.variable = Variable(result_type)
        return node

    def visit_Return(self, node):
        value = self.visit(node.value)
        type = value.variable.type

        if self.return_variable.type is None:
            self.return_variable.type = type
            node.value = value
        else:
            # todo: in case of unpromotable types, return object?
            self.return_variable.type = self.promote_types(self.return_type, type)
            node.value = nodes.DeferredCoercionNode(value, self.return_variable)

        return node

    #
    ### Unsupported nodes
    #

    def visit_Global(self, node):
        raise NotImplementedError("Global keyword")

class TypeSettingVisitor(visitors.NumbaVisitor):
    """
    Set node.type for all AST nodes after type inference from node.variable.
    Allows for deferred coercions (may be removed in the future).
    """

    def visit(self, node):
        if isinstance(node, ast.expr):
            node.type = node.variable.type
        super(TypeSettingVisitor, self).visit(node)
