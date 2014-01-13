# -*- coding: utf-8 -*-
"""
This module provides a variety of transforms that transform the AST
into a final form ready for code generation.

Below follows an explanation and justification of the design of the main
compilation stages in numba.

We start with a Python AST, compiled from source code or decompiled from
bytecode using meta. We run the following transformations:

    1) Type inference

        Infer types of all expressions, and fix the types of all local
        variables. Local variable types are promoted (for instance float
        to double), but cannot change (e.g. string cannot be assigned to
        float).

        When the type inferencer cannot determine a type, such as when it
        calls a Python function or method that is not a Numba function, it
        assumes type object. Object variables may be coerced to and from
        most native types.

        The type inferencer inserts CoercionNode nodes that perform such
        coercions, as well as coercions between promotable native types.
        It also resolves the return type of many math functions called
        in the numpy, math and cmath modules.

        Each AST expression node has a Variable that holds the type of
        the expression, as well as any meta-data such as constant values
        that have been determined.

    2) Transform for loops

        Provides some transformations of for loops over arrays to loops
        over a range. Iteration over range/xrange is resolved at
        compilation time.

        What I would like to see is the generation of a RangeNode holding
        a ast.Compare and an iteration variable incrementing ast.BinOp.

    3) Low level specializations (LateSpecializer)

        This stage performs low-level specializations. For instance it
        resolves coercions to and from object into calls such as
        PyFloat_FromDouble, with a fallback to Py_BuildValue/PyArg_ParseTuple.

        This specializer also has the responsibility to ensure that new
        references are accounted for by refcounting ObjectTempNode nodes.
        This node absorbs the references and lets parent nodes borrow the
        reference. At function cleanup, it decrefs its value. In loops,
        it also decrefs any previous value, if set. Hence, these temporaries
        must be initialized to NULL.

        An object temporary is specific to one specific sub-expression, and
        they are not reused (like in Cython).

        It also rewrites object attribute access and method calls into
        PyObject_GetAttrString etc.

    4) Code generation

        Generate LLVM code from the transformed AST.

        This should be as minimal as possible, and should *not* contain
        blobs of code performing complex operations. Instead, complex
        operations should be broken down by AST transformers into
        fundamental operations that are already supported by the AST.

        This way we maximize code reuse, and make potential future additions
        of different code generation backends easier. This can be taken
        only so far, since low-level transformations must also tailor to
        limitations of the code generation backend, such as intrinsic LLVM
        calls or calls into libc. However, code reuse is especially convenient
        in the face of type coercions, which LLVM does not provide any
        leniency for.
"""
from __future__ import print_function, division, absolute_import


import sys
import ast
import ctypes
import warnings

if __debug__:
    import pprint

import numba
from numba import *
from numba import error
from .minivect import codegen
from numba import macros, utils, typesystem
from numba import visitors, nodes
from numba import function_util
from numba.typesystem import is_obj, promote_to_native
from numba.type_inference.modules import mathmodule
from numba.nodes import constnodes
from numba.external import utility
from numba.utils import dump

import llvm.core
import numpy as np

logger = logging.getLogger(__name__)

from numba.external import pyapi

# ______________________________________________________________________

def get_funcname(py_func):
    if py_func in (abs, np.abs):
        return 'abs'
    elif py_func is np.round:
        return 'round'

    return mathmodule.ufunc2math.get(py_func.__name__, py_func.__name__)

def resolve_pow(env, restype, args):
    promote = env.crnt.typesystem.promote
    if restype.is_numeric:
        type = reduce(promote, [double, restype] + [a.type for a in args])
        signature = type(*[type] * len(args))
        result = nodes.MathCallNode(signature, args, None, name='pow')
    else:
        result = nodes.call_pyfunc(pow, args)
    return nodes.CoercionNode(result, restype)

def math_call(env, name, args, dst_type):
    signature = dst_type(*[a.type for a in args])
    return nodes.MathCallNode(signature, args, None, name=name)

def math_call2(env, name, call_node):
    return math_call(env, name, [call_node.args[0]], call_node.type)

# ______________________________________________________________________

class BuiltinResolver(object):
    """
    Perform final low-level transformations such as abs(value) -> fabs(value)
    """

    def __init__(self, env):
        self.env = env
        self.external_call = partial(function_util.external_call,
                                     self.env.context,
                                     self.env.crnt.llvm_module)

    def resolve_builtin_call(self, node, func):
        """
        Resolve an ast.Call() of a built-in function.

        Returns None if no specific transformation is applied.
        """
        resolver = getattr(self, '_resolve_' + func.__name__, None)
        if resolver is not None:
            # Pass in the first argument type
            argtype = None
            if len(node.args) >= 1:
                argtype = node.args[0].variable.type

            return resolver(func, node, argtype)

        return None

    def resolve_builtin_call_or_object(self, node, func):
        """
        Resolve an ast.Call() of a built-in function, or call the built-in
        through the object layer otherwise.
        """
        result = self.resolve_builtin_call(node, func)
        if result is None:
            result = nodes.call_pyfunc(func, node.args)

        return nodes.CoercionNode(result, node.type)


    def _resolve_abs(self, func, node, argtype):
        if argtype.is_int and not argtype.signed:
            # abs() on unsigned integral value
            return node.args[0]
        elif not node.type.is_numeric:
            result = nodes.call_pyfunc(func, node.args)
        else:
            return math_call2(self.env, 'abs', node)

    def _resolve_round(self, func, node, argtype):
        return nodes.call_pyfunc(round, node.args)

    def _resolve_pow(self, func, node, argtype):
        return resolve_pow(self.env, node.type, node.args)

    def _resolve_int_number(self, func, node, argtype, dst_type, ext_name):
        assert len(node.args) == 2

        arg1, arg2 = node.args
        if arg1.variable.type.is_string:
            return nodes.CoercionNode(
                nodes.ObjectTempNode(
                    self.external_call(ext_name, args=[arg1, nodes.NULL, arg2])),
                dst_type=dst_type)

    def _resolve_int(self, func, node, argtype, dst_type=int_):
        if PY3:
            return self._resolve_int_number(func, node, argtype, long_, 'PyLong_FromString')
        return self._resolve_int_number(func, node, argtype, int_, 'PyInt_FromString')

    def _resolve_long(self, func, node, argtype, dst_type=int_):
        return self._resolve_int_number(func, node, argtype, long_, 'PyLong_FromString')

    def _resolve_len(self, func, node, argtype):
        if argtype.is_string:
            call = self.external_call('strlen', node.args)
            return call # nodes.CoercionNode(call, Py_ssize_t)

class ResolveCoercions(visitors.NumbaTransformer):

    def visit_CoercionNode(self, node):
        if not isinstance(node, nodes.CoercionNode):
            # CoercionNode.__new__ returns the node to be coerced if it doesn't
            # need coercion
            return node

        node_type = node.node.type
        dst_type = node.dst_type
        if __debug__ and self.env and self.env.debug_coercions:
            logger.debug('coercion: %s --> %s\n%s',
                         node_type, dst_type, utils.pformat_ast(node))

        # TODO: the below is a problem due to implicit string <-> int coercions!
        if (node_type.is_string and dst_type.is_numeric and not
            (node_type.is_pointer or node_type.is_null)):
            if dst_type.typename in ('char', 'uchar'):
                raise error.NumbaError(
                    node, "Conversion from string to (u)char not yet supported")
            result = self.str_to_int(dst_type, node)
        elif self.nopython and (is_obj(node_type) ^ is_obj(dst_type)):
            raise error.NumbaError(node, "Cannot coerce to or from object in "
                                         "nopython context")
        elif is_obj(node.dst_type) and not is_obj(node_type):
            node = nodes.ObjectTempNode(nodes.CoerceToObject(
                    node.node, node.dst_type, name=node.name))
            result = self.visit(node)
        elif is_obj(node_type) and not is_obj(node.dst_type):
            node = nodes.CoerceToNative(node.node, node.dst_type,
                                        name=node.name)
            result = self.visit(node)
        elif node_type.is_null:
            if not dst_type.is_pointer:
                raise error.NumbaError(node.node,
                                       "NULL must be cast or implicitly "
                                       "coerced to a pointer type")
            result = self.visit(nodes.NULL.coerce(dst_type))
        elif node_type.is_numeric and dst_type.is_bool:
            to_bool = ast.Compare(node.node, [ast.NotEq()],
                                  [nodes.const(0, node_type)])
            to_bool = nodes.typednode(to_bool, bool_)
            result = self.visit(to_bool)
        else:
            self.generic_visit(node)

            if dst_type == node.node.type:
                result = node.node
            else:
                result = node

        if __debug__ and self.env and self.env.debug_coercions:
            logger.debug('result = %s', utils.pformat_ast(result))

        return result

    def str_to_int(self, dst_type, node):
        # TODO: int <-> string conversions are explicit, this should not
        # TODO: be a coercion
        if self.nopython:
            node = nodes.CoercionNode(
                function_util.external_call(
                    self.context,
                    self.llvm_module,
                    ('atol' if dst_type.is_int else 'atof'),
                    args=[node.node]),
                dst_type, name=node.name, )
        else:
            if dst_type.is_int:
                cvtobj = function_util.external_call(
                    self.context,
                    self.llvm_module,
                    'PyInt_FromString' if not PY3 else 'PyLong_FromString',
                    args=[node.node, nodes.NULL,
                          nodes.const(10, int_)])
            else:
                cvtobj = function_util.external_call(
                    self.context,
                    self.llvm_module,
                    'PyFloat_FromString',
                    args=[node.node,
                          nodes.const(0, Py_ssize_t)])
            node = nodes.CoerceToNative(nodes.ObjectTempNode(cvtobj),
                                        dst_type, name=node.name)
        result = self.visit(node)
        return result

    def convert_int_to_object(self, arg):
        funcs = ["__Numba_PyInt_FromLongLong",
                 "__Numba_PyInt_FromUnsignedLongLong"]
        func = funcs[arg.type.signed]
        return function_util.utility_call(self.context, self.llvm_module,
                                          func, [arg])

    def visit_CoerceToObject(self, node):
        new_node = node

        node_type = node.node.type
        if node_type.is_bool:
            new_node = function_util.external_call(self.context,
                                                   self.llvm_module,
                                                   "PyBool_FromLong",
                                                   args=[node.node])
        elif node_type.is_numeric and node_type.typename not in ('char', 'uchar'):
            cls = None
            args = node.node,
            if node_type.is_int:
                new_node = self.convert_int_to_object(node.node)
            elif node_type.is_float:
                cls = pyapi.PyFloat_FromDouble
            elif node_type.is_complex:
                cls = pyapi.PyComplex_FromDoubles
                complex_value = nodes.CloneableNode(node.node)
                args = [
                    nodes.ComplexAttributeNode(complex_value, "real"),
                    nodes.ComplexAttributeNode(complex_value.clone, "imag")
                ]
            elif node_type.is_numpy_datetime:
                datetime_value = nodes.CloneableNode(node.node)
                args = [
                    nodes.DateTimeAttributeNode(datetime_value, 'timestamp'),
                    nodes.DateTimeAttributeNode(datetime_value.clone, 'units'),
                    nodes.ConstNode(np.datetime64(), object_),
                ]
                new_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        "create_numpy_datetime", args=args)
            elif node_type.is_datetime:
                datetime_value = nodes.CloneableNode(node.node)
                args = [
                    nodes.DateTimeAttributeNode(datetime_value, 'timestamp'),
                    nodes.DateTimeAttributeNode(datetime_value.clone, 'units'),
                ]
                new_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        "create_python_datetime", args=args)
            elif node_type.is_timedelta:
                timedelta_value = nodes.CloneableNode(node.node)
                args = [
                    nodes.TimeDeltaAttributeNode(timedelta_value, 'diff'),
                    nodes.TimeDeltaAttributeNode(timedelta_value.clone, 'units'),
                    nodes.ConstNode(np.timedelta64(), object_),
                ]
                new_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        "create_numpy_timedelta", args=args)
            else:
                raise error.NumbaError(
                    node, "Don't know how to coerce type %r to PyObject" %
                    node_type)

            if cls:
                new_node = function_util.external_call(self.context,
                                                       self.llvm_module,
                                                       cls.__name__,
                                                       args=args)
        elif node_type.is_pointer and not node_type in (char.pointer(), string_):
            # Create ctypes pointer object
            ctypes_pointer_type = node_type.to_ctypes()
            args = [nodes.CoercionNode(node.node, int64),
                    nodes.ObjectInjectNode(ctypes_pointer_type, object_)]
            new_node = nodes.call_pyfunc(ctypes.cast, args)

        self.generic_visit(new_node)
        return new_node

    def object_to_int(self, node, dst_type):
        """
        Return node that converts the given node to the dst_type.
        This also performs overflow/underflow checking, and conversion to
        a Python int or long if necessary.

        PyLong_AsLong and friends do not do this (overflow/underflow checking
        is only for longs, and conversion to int|long depends on the Python
        version).
        """
        dst_type = promote_to_native(dst_type)
        assert dst_type in utility.object_to_numeric, (dst_type, utility.object_to_numeric)
        utility_func = utility.object_to_numeric[dst_type]
        result = function_util.external_call_func(self.context,
                                                  self.llvm_module,
                                                  utility_func,
                                                  args=[node])
        return result

    def coerce_to_function_pointer(self, node, jit_func_type, func_pointer_type):
        jit_func = jit_func_type.jit_func
        if jit_func.signature != func_pointer_type.base_type:
            raise error.NumbaError(node,
                                   "Cannot coerce jit funcion %s to function of type %s" % (
                                       jit_func, func_pointer_type))
        pointer = self.env.llvm_context.get_pointer_to_function(jit_func.lfunc)
        new_node = nodes.const(pointer, func_pointer_type)
        return new_node

    def visit_CoerceToNative(self, node):
        """
        Try to perform fast coercion using e.g. PyLong_AsLong(), with a
        fallback to PyArg_ParseTuple().
        """
        new_node = None

        from_type = node.node.type
        node_type = node.type

        if node_type.is_numeric:
            cls = None
            if node_type == size_t:
                node_type = ulonglong

            if node_type.is_int: # and not
                new_node = self.object_to_int(node.node, node_type)
            elif node_type.is_float:
                cls = pyapi.PyFloat_AsDouble
            elif node_type.is_complex:
                # FIXME: This conversion has to be pretty slow.  We
                # need to move towards being ABI-savvy enough to just
                # call PyComplex_AsCComplex().
                cloneable = nodes.CloneableNode(node.node)
                new_node = nodes.ComplexNode(
                    real=function_util.external_call(
                        self.context, self.llvm_module,
                        "PyComplex_RealAsDouble", args=[cloneable]),
                    imag=function_util.external_call(
                        self.context, self.llvm_module,
                        "PyComplex_ImagAsDouble", args=[cloneable.clone]))
            elif node_type.is_numpy_datetime:
                timestamp_func = function_util.utility_call(
                    self.context, self.llvm_module,
                    "convert_numpy_datetime_to_timestamp", args=[node.node])
                units_func = function_util.utility_call(
                    self.context, self.llvm_module,
                    "convert_numpy_datetime_to_units", args=[node.node])
                new_node = nodes.DateTimeNode(timestamp_func, units_func)
            elif node_type.is_datetime:
                timestamp_func = function_util.utility_call(
                    self.context, self.llvm_module,
                    "pydatetime2timestamp", args=[node.node])
                units_func = function_util.utility_call(
                    self.context, self.llvm_module,
                    "pydatetime2units", args=[node.node])
                new_node = nodes.DateTimeNode(timestamp_func, units_func)
            elif node_type.is_numpy_timedelta:
                diff_func = function_util.utility_call(
                    self.context, self.llvm_module,
                    "convert_numpy_timedelta_to_diff", args=[node.node])
                units_func = function_util.utility_call(
                    self.context, self.llvm_module,
                    "convert_numpy_timedelta_to_units", args=[node.node])
                new_node = nodes.DateTimeNode(diff_func, units_func)
            elif node_type.is_timedelta:
                raise NotImplementedError
            else:
                raise error.NumbaError(
                    node, "Don't know how to coerce a Python object to a %r" %
                    node_type)

            if cls:
                # TODO: error checking!
                new_node = function_util.external_call(self.context,
                                                       self.llvm_module,
                                                       cls.__name__,
                                                       args=[node.node])
        elif node_type.is_pointer and not node_type.is_string:
            if from_type.is_jit_function and node_type.base_type.is_function:
                new_node = self.coerce_to_function_pointer(
                    node, from_type, node_type)
            else:
                raise error.NumbaError(node, "Obtaining pointers from objects "
                                             "is not yet supported (%s)" % node_type)
        elif node_type.is_void:
            raise error.NumbaError(node, "Cannot coerce %s to void" %
                                   (from_type,))

        if new_node is None:
            # Create a tuple for PyArg_ParseTuple
            new_node = node
            new_node.node = ast.Tuple(elts=[node.node], ctx=ast.Load())
            self.generic_visit(node)
            return node

        if new_node.type != node.type:
            # Fast native coercion. E.g. coercing an object to an int_
            # will use PyLong_AsLong, but that will return a long_. We
            # need to coerce the long_ to an int_
            new_node = nodes.CoercionNode(new_node, node.type)

        # Specialize replacement node
        new_node = self.visit(new_node)
        return new_node


class LateSpecializer(ResolveCoercions,
                      visitors.NoPythonContextMixin):

    def visit_FunctionDef(self, node):
        self.builtin_resolver = BuiltinResolver(self.env)
        node.decorator_list = self.visitlist(node.decorator_list)

        # Make sure to visit the entry block (not part of the CFG) and the
        # first actual code block which may have synthetically
        # inserted promotions
        self.visit_ControlBlock(node.flow.blocks[0])
        self.visit_ControlBlock(node.flow.blocks[1])

        node.body = self.visitlist(node.body)

        ret_type = self.func_signature.return_type
        self.verify_context(ret_type)

        self.setup_error_return(node, ret_type)
        return node

    def verify_context(self, ret_type):
        if ret_type.is_object or ret_type.is_array:
            # This will require some increfs, but allow it if people
            # use 'with python' later on. If 'with python' isn't used, a
            # return will issue the error
            #if self.nopython:
            #    raise error.NumbaError(
            #            node, "Function cannot return object in "
            #                  "nopython context")
            pass

    def setup_error_return(self, node, ret_type):
        """
        Set FunctionDef.error_return to the AST statement that returns a
        "bad value" that can be used as error indicator.
        """
        value = nodes.badval(ret_type)

        if value is not None:
            value = nodes.CoercionNode(value, dst_type=ret_type).cloneable

        error_return = ast.Return(value=value)

        if self.nopython and is_obj(self.func_signature.return_type):
            error_return = nodes.WithPythonNode(body=[error_return])

        error_return = self.visit(error_return)
        node.error_return = error_return

    def visit_ControlBlock(self, node):
        # print node
        self.visitchildren(node)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        return node

    def check_context(self, node):
        if self.nopython:
            raise error.NumbaError(node, "Cannot construct object in "
                                         "nopython context")

    def _print_nopython(self, value, dest=None):
        if dest is not None:
            raise error.NumbaError(dest, "No file may be given in nopython mode")

        # stdin, stdout, stderr = stdio_util.get_stdio_streams()
        # stdout = stdio_util.get_stream_as_node(stdout)

        format = codegen.get_printf_specifier(value.type)
        if format is None:
            raise error.NumbaError(
                value, "Printing values of type '%s' is not supported "
                       "in nopython mode" % (value.type,))

        return function_util.external_call(
                                       self.context,
                                       self.llvm_module,
                                       'printf',
                                       args=[nodes.const(format, c_string_type),
                                             value])


    def _print(self, value, dest=None):
        signature, lfunc = self.context.external_library.declare(
                                                         self.llvm_module,
                                                         'PyObject_CallMethod')

        if dest is None:
            dest = nodes.ObjectInjectNode(sys.stdout)

        value = function_util.external_call(self.context,
                                            self.llvm_module,
                                           "PyObject_Str",
                                            args=[value])
        args = [dest, nodes.ConstNode("write"), nodes.ConstNode("O"), value]
        return nodes.NativeCallNode(signature, args, lfunc)

    def visit_Print(self, node):
        if self.nopython:
            printfunc = self._print_nopython
            dst_type = string_
        else:
            printfunc = self._print
            dst_type = object_

        result = []

        if node.values:
            print_space = printfunc(nodes.const(" ", dst_type), node.dest)
            for value in node.values:
                result.append(printfunc(value, node.dest))
                result.append(print_space)

            if node.nl:
                result.pop() # pop last space

        if node.nl:
            result.append(printfunc(nodes.const("\n", dst_type), node.dest))

        return ast.Suite(body=self.visitlist(result))

    def visit_Tuple(self, node):
        self.check_context(node)

        sig, lfunc = self.context.external_library.declare(self.llvm_module,
                                                           'PyTuple_Pack')
        objs = self.visitlist(nodes.CoercionNode.coerce(node.elts, object_))
        n = nodes.ConstNode(len(node.elts), Py_ssize_t)
        args = [n] + objs
        new_node = nodes.NativeCallNode(sig, args, lfunc, name='tuple')
        # TODO: determine element type of node.elts
        new_node.type = typesystem.tuple_(object_, size=len(node.elts))
        return nodes.ObjectTempNode(new_node)

    def visit_List(self, node):
        self.check_context(node)
        self.generic_visit(node)
        return nodes.ObjectTempNode(node)

    def visit_Dict(self, node):
        self.check_context(node)
        self.generic_visit(node)
        return nodes.ObjectTempNode(node)

    def visit_ObjectCallNode(self, node):
        # self.generic_visit(node)
        assert node.function

        if self.nopython:
            meth_name = node.name and ' (%r)' % node.name
            raise error.NumbaError(node, "Cannot use object call in "
                                         "nopython context" + meth_name)

        node.function = self.visit(node.function)
        node.args_tuple = self.visit(node.args_tuple)
        node.kwargs_dict = self.visit(node.kwargs_dict)
        return nodes.ObjectTempNode(node)

    def visit_Call(self, node):
        func_type = node.func.type

        if self.query(node, "is_math") and node.type.is_numeric:
            assert node.func.type.is_known_value
            name = get_funcname(node.func.type.value)
            result = math_call(self.env, name, node.args, node.type)

        elif func_type.is_builtin:
            result = self.builtin_resolver.resolve_builtin_call_or_object(
                node, func_type.func)

        else:
            result = nodes.call_obj(node)

        return self.visit(result)

    def _c_string_slice(self, node):
        ret_val = node
        logger.debug(node.slice)
        node_slice = node.slice
        if isinstance(node_slice, nodes.ObjectInjectNode):
            node_slice = node.slice.object
            lower, upper, step = (
                value if value is None else nodes.const(value, size_t)
                for value in (node_slice.start, node_slice.stop,
                              node_slice.step))
        else:
            lower, upper, step = (node_slice.lower, node_slice.upper,
                                  node_slice.step)
        if step is None:
            node_value = self.visit(node.value)
            if lower is None:
                lower = nodes.const(0, size_t)
            if upper is None:
                ret_val = nodes.LLMacroNode(
                    macros.c_string_slice_1.__signature__,
                    macros.c_string_slice_1, self.visit(node.value),
                    self.visit(lower))
            else:
                ret_val = nodes.LLMacroNode(
                    macros.c_string_slice_2.__signature__,
                    macros.c_string_slice_2, self.visit(node.value),
                    self.visit(lower), self.visit(upper))
            logger.debug(ret_val)
        else:
            raise NotImplementedError('String slices where step != None.')
        return ret_val

    def visit_Subscript(self, node):
        if isinstance(node.value, nodes.ArrayAttributeNode):
            if node.value.is_read_only and isinstance(node.ctx, ast.Store):
                raise error.NumbaError("Attempt to load read-only attribute")

        # Short-circuit visiting a Slice child if this is a nopython
        # string slice.
        if (self.nopython and node.value.type.is_string and
                node.type.is_string):
            return self.visit(self._c_string_slice(node))

        # logging.debug(ast.dump(node))
        # TODO: do this in the respective cases below when needed
        self.generic_visit(node)

        node_type = node.value.type
        if ((node_type.is_object and not node_type.is_array) or
            (node_type.is_array and node.slice.type.is_object)):
            # Array or object slicing
            if isinstance(node.ctx, ast.Load):
                result = function_util.external_call(self.context,
                                                     self.llvm_module,
                                                     'PyObject_GetItem',
                                                     args=[node.value,
                                                           node.slice])
                node = nodes.CoercionNode(result, dst_type=node.type)
                node = self.visit(node)
            else:
                # This is handled in visit_Assign
                pass
        elif (node.value.type.is_array and node.type.is_numpy_datetime and
                node.slice.type.is_int):
            # JNB: ugly hack to make array of datetimes look like array of
            # int64, since numba datetime type doesn't match numpy datetime type.
            node.value.type = array_(int64, node.value.type.ndim,
                node.value.type.is_c_contig,
                node.value.type.is_f_contig,
                node.value.type.inner_contig)
            node.value.variable.type = node.value.type
            data_node = nodes.DataPointerNode(node.value, node.slice, node.ctx)

            units_node = function_util.utility_call(
                    self.context, self.llvm_module,
                    "get_units_num",
                    args=[nodes.ConstNode(node_type.dtype.units_char, string_)])
            node = nodes.DateTimeNode(data_node, units_node)
        elif (node.value.type.is_array and node.type.is_numpy_timedelta and
                node.slice.type.is_int):
            # JNB: ugly hack to make array of timedeltas look like array of
            # int64, since numba timedelta type doesn't match numpy timedelta type.
            node.value.type = array_(int64, node.value.type.ndim,
                node.value.type.is_c_contig,
                node.value.type.is_f_contig,
                node.value.type.inner_contig)
            node.value.variable.type = node.value.type
            data_node = nodes.DataPointerNode(node.value, node.slice, node.ctx)

            units_node = function_util.utility_call(
                    self.context, self.llvm_module,
                    "get_units_num",
                    args=[nodes.ConstNode(node_type.dtype.units_char, string_)])
            node = nodes.TimeDeltaNode(data_node, units_node)
        elif (node.value.type.is_array and not node.type.is_array and
                  node.slice.type.is_int):
            # Array index with integer indices
            node = nodes.DataPointerNode(node.value, node.slice, node.ctx)
        elif node.value.type.is_string and node.type.is_string:
            node.value = nodes.CoercionNode(node.value, dst_type = object_)
            node.type = object_
            node = nodes.CoercionNode(nodes.ObjectTempNode(node),
                                      dst_type = c_string_type)
            node = self.visit(node)

        return node

    def visit_ExtSlice(self, node):
        if node.type.is_object:
            return self.visit(ast.Tuple(elts=node.dims, ctx=ast.Load()))
        else:
            if node.type.is_float:
                self.warn(node, "Using a float for indexing")
            self.generic_visit(node)
            return node

    def visit_Index(self, node):
        return self.visit(node.value)

    def allocate_struct_on_stack(self, assmnt_node, target):
        # Allocate struct on stack
        temp = nodes.TempNode(target.type)
        assmnt_node.targets[0] = temp.store()
        assmnt_node.value = self.visit(assmnt_node.value)

        # Expose LLVM value through SSA (patch the Variable or the
        # LHS). We need to store the pointer to the struct (the alloca)
        ssa_assmnt = ast.Assign(targets=[target], value=temp.store())

        return ast.Suite(body=[assmnt_node, ssa_assmnt])

    def visit_Assign(self, node):
        target = node.targets[0]
        target_is_subscript = (len(node.targets) == 1 and
                               isinstance(target, ast.Subscript))
        if target_is_subscript and is_obj(target.type):
            # Slice assignment / index assignment w/ objects
            # TODO: discount array indexing with dtype object
            target = self.visit(target)
            obj = target.value
            key = target.slice
            value = self.visit(node.value)
            call = function_util.external_call(self.context,
                                               self.llvm_module,
                                               'PyObject_SetItem',
                                               args=[obj, key, value])
            return self.visit(call)

        elif target.type.is_struct and nodes.is_name(target):
            node = self.allocate_struct_on_stack(node, target)
            return node

        self.generic_visit(node)
        return node

    def visit_Slice(self, node):
        """
        Rewrite slice objects. Do this late in the pipeline so that other
        code can still recognize the code structure.
        """
        slice_values = [node.lower, node.upper, node.step]

        if self.nopython:
            raise error.NumbaError(node, "Cannot slice in nopython context")

        if node.variable.is_constant:
            return self.visit(nodes.ObjectInjectNode(node.variable.constant_value))

        bounds = []
        for node in slice_values:
            if node is None:
                bounds.append(nodes.NULL_obj)
            else:
                bounds.append(node)

        new_slice = function_util.external_call(self.context,
                                                self.llvm_module,
                                                'PySlice_New',
                                                args=bounds,
                                                temp_name='slice')
        return self.visit(new_slice)
        # return nodes.ObjectTempNode(new_slice)

    def visit_Attribute(self, node):
        if (self.nopython and not node.value.type.is_module and
                not node.value.type.is_complex and
                not node.value.type.is_datetime and
                not node.value.type.is_timedelta):
            raise error.NumbaError(
                    node, "Cannot access Python attribute in nopython context (%s)" % node.attr)

        if node.value.type.is_complex:
            value = self.visit(node.value)
            return nodes.ComplexAttributeNode(value, node.attr)
        elif node.value.type.is_numpy_datetime:
            value = self.visit(node.value)
            if node.attr in ['year', 'month', 'day', 'hour', 'min', 'sec']:
                func_dict = {'year' : 'extract_datetime_year',
                             'month' : 'extract_datetime_month',
                             'day' : 'extract_datetime_day',
                             'hour' : 'extract_datetime_hour',
                             'min' : 'extract_datetime_min',
                             'sec' : 'extract_datetime_sec',}
                value = nodes.CloneableNode(value)
                timestamp_node = nodes.DateTimeAttributeNode(value,
                    'timestamp')
                unit_node = nodes.DateTimeAttributeNode(value.clone, 'units')
                new_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        func_dict[node.attr],
                        args=[timestamp_node, unit_node])
                return new_node
            else:
                return nodes.DateTimeAttributeNode(value, node.attr)

        elif node.value.type.is_datetime:
            value = self.visit(node.value)
            return nodes.DateTimeAttributeNode(value, node.attr)
        elif node.value.type.is_timedelta:
            value = self.visit(node.value)
            return nodes.TimeDeltaAttributeNode(value, node.attr)
        elif node.type.is_numpy_attribute:
            return nodes.ObjectInjectNode(node.type.value)
        elif node.type.is_numpy_dtype:
            dtype_type = node.type.dtype
            return nodes.ObjectInjectNode(dtype_type.get_dtype())
        elif is_obj(node.value.type):
            if node.value.type.is_module:
                # Resolve module attributes as constants
                if node.type.is_module_attribute:
                    new_node = nodes.ObjectInjectNode(node.type.value)
                else:
                    new_node = nodes.ConstNode(getattr(node.value.type.module,
                                                       node.attr))
            else:
                new_node = function_util.external_call(
                                        self.context,
                                        self.llvm_module,
                                        'PyObject_GetAttrString',
                                        args=[node.value,
                                              nodes.ConstNode(node.attr)])
            return self.visit(new_node)

        self.generic_visit(node)
        return node

    def visit_ArrayNewNode(self, node):
        if self.nopython:
            raise error.NumbaError(
                node, "Cannot yet allocate new array in nopython context")

        PyArray_Type = nodes.ObjectInjectNode(np.ndarray)
        descr = nodes.ObjectInjectNode(node.type.dtype.get_dtype()).cloneable
        ndim = nodes.const(node.type.ndim, int_)
        flags = nodes.const(0, int_)
        args = [PyArray_Type, descr.clone, ndim,
                node.shape, node.strides, node.data, flags]

        incref_descr = nodes.IncrefNode(descr)
        incref_base = None
        setbase = None

        if node.base is None:
            args.append(nodes.NULL_obj)
        else:
            base = nodes.CloneableNode(node.base)
            incref_base = nodes.IncrefNode(base)
            args.append(base.clone)

        array = nodes.PyArray_NewFromDescr(args)
        array = nodes.ObjectTempNode(array).cloneable
        body = [incref_descr, incref_base, array, setbase]

        if node.base is not None:
            body.append(nodes.PyArray_SetBaseObject([array.clone, base.clone]))

        # TODO: PyArray_UpdateFlags()
        result = nodes.ExpressionNode(filter(None, body), array.clone)
        return self.visit(result)

    def visit_ArrayNewEmptyNode(self, node):
        if self.nopython:
            raise error.NumbaError(
                node, "Cannot yet allocate new empty array in nopython context")

        ndim = nodes.const(node.type.ndim, int_)
        dtype = nodes.const(node.type.dtype.get_dtype(), object_).cloneable
        is_fortran = nodes.const(node.is_fortran, int_)
        result = nodes.PyArray_Empty([ndim, node.shape, dtype, is_fortran])
        result = nodes.ObjectTempNode(result)
        incref_descr = nodes.IncrefNode(dtype)
        return self.visit(nodes.ExpressionNode([incref_descr], result))

    def visit_Name(self, node):
        if node.variable.is_constant:
            obj = node.variable.constant_value
            return self.visit(nodes.const(obj, node.type))

        return node

    def visit_Return(self, node):
        return_type = self.func_signature.return_type
        if node.value is not None:
            node.value = self.visit(nodes.CoercionNode(node.value, return_type))
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        return node

    def _object_binop(self, node, api_name):
        return self.visit(
            function_util.external_call(self.context,
                                        self.llvm_module,
                                        api_name,
                                        args=[node.left,
                                              node.right]))

    def _object_Add(self, node):
        return self._object_binop(node, 'PyNumber_Add')

    def _object_Sub(self, node):
        return self._object_binop(node, 'PyNumber_Subtract')

    def _object_Mult(self, node):
        return self._object_binop(node, 'PyNumber_Multiply')

    def _object_Div(self, node):
        if PY3:
            return self._object_binop(node, 'PyNumber_TrueDivide')
        else:
            return self._object_binop(node, 'PyNumber_Divide')

    def _object_Mod(self, node):
        return self._object_binop(node, 'PyNumber_Remainder')

    def _object_Pow(self, node):
        args = [node.left,
                node.right,
                nodes.ObjectInjectNode(None)]
        return self.visit(function_util.external_call(self.context,
                                                       self.llvm_module,
                                                       'PyNumber_Power',
                                                       args=args),
                            llvm_module=self.llvm_module)

    def _object_LShift(self, node):
        return self._object_binop(node, 'PyNumber_Lshift')

    def _object_RShift(self, node):
        return self._object_binop(node, 'PyNumber_Rshift')

    def _object_BitOr(self, node):
        return self._object_binop(node, 'PyNumber_Or')

    def _object_BitXor(self, node):
        return self._object_binop(node, 'PyNumber_Xor')

    def _object_BitAnd(self, node):
        return self._object_binop(node, 'PyNumber_And')

    def _object_FloorDiv(self, node):
        return self._object_binop(node, 'PyNumber_FloorDivide')

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            return self.visit(resolve_pow(self.env, node.type, [node.left,
                                                                node.right]))

        self.generic_visit(node)
        if is_obj(node.left.type) or is_obj(node.right.type):
            op_name = type(node.op).__name__
            op_method = getattr(self, '_object_%s' % op_name, None)
            if op_method:
                node = op_method(node)
            else:
                raise error.NumbaError(
                    node, 'Unsupported binary operation for object: %s' %
                    op_name)
        elif node.left.type.is_datetime and node.right.type.is_datetime:
            if isinstance(node.op, ast.Sub):
                datetime_value = nodes.CloneableNode(node.left)
                units1_node = nodes.DateTimeAttributeNode(
                    datetime_value, 'units')
                datetime_value = nodes.CloneableNode(node.right)
                units2_node = nodes.DateTimeAttributeNode(
                    datetime_value, 'units')

                unit_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        "get_target_unit_for_datetime_datetime",
                        args=[units1_node, units2_node])

                datetime_value = nodes.CloneableNode(node.left)
                args1 = [
                    nodes.DateTimeAttributeNode(datetime_value, 'timestamp'),
                    nodes.DateTimeAttributeNode(datetime_value.clone, 'units'),
                ]
                datetime_value = nodes.CloneableNode(node.right)
                args2 = [
                    nodes.DateTimeAttributeNode(datetime_value, 'timestamp'),
                    nodes.DateTimeAttributeNode(datetime_value.clone, 'units'),
                ]

                diff_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        "sub_datetime_datetime", args=args1+args2+[unit_node])

                node = nodes.TimeDeltaNode(diff_node, unit_node)
            else:
                raise NotImplementedError
        elif (node.left.type.is_datetime and
              node.right.type.is_timedelta) or \
             (node.left.type.is_timedelta and
              node.right.type.is_datetime):
            if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
                datetime_value = nodes.CloneableNode(node.left)
                if node.left.type.is_datetime:
                    units1_node = nodes.DateTimeAttributeNode(
                        datetime_value, 'units')
                else:
                    units1_node = nodes.TimeDeltaAttributeNode(
                        datetime_value, 'units')

                datetime_value = nodes.CloneableNode(node.right)
                if node.right.type.is_datetime:
                    units2_node = nodes.DateTimeAttributeNode(
                        datetime_value, 'units')
                else:
                    units2_node = nodes.TimeDeltaAttributeNode(
                        datetime_value, 'units')

                unit_node = function_util.utility_call(
                        self.context, self.llvm_module,
                        "get_target_unit_for_datetime_timedelta",
                        args=[units1_node, units2_node])

                datetime_value = nodes.CloneableNode(node.left)
                if node.left.type.is_datetime:
                    args1 = [
                        nodes.DateTimeAttributeNode(
                            datetime_value, 'timestamp'),
                        nodes.DateTimeAttributeNode(
                            datetime_value.clone, 'units'),]
                else:
                    args1 = [
                        nodes.TimeDeltaAttributeNode(
                            datetime_value, 'diff'),
                        nodes.TimeDeltaAttributeNode(
                            datetime_value.clone, 'units'),]
                datetime_value = nodes.CloneableNode(node.right)
                if node.right.type.is_datetime:
                    args2 = [
                        nodes.DateTimeAttributeNode(
                            datetime_value, 'timestamp'),
                        nodes.DateTimeAttributeNode(
                            datetime_value.clone, 'units'),]
                else:
                    args2 = [
                        nodes.TimeDeltaAttributeNode(
                            datetime_value, 'diff'),
                        nodes.TimeDeltaAttributeNode(
                            datetime_value.clone, 'units'),]

                if isinstance(node.op, ast.Add):
                    diff_node = function_util.utility_call(
                            self.context, self.llvm_module,
                            "add_datetime_timedelta",
                            args=args1+args2+[unit_node])
                elif isinstance(node.op, ast.Sub):
                    diff_node = function_util.utility_call(
                            self.context, self.llvm_module,
                            "sub_datetime_timedelta",
                            args=args1+args2+[unit_node])

                node = nodes.DateTimeNode(diff_node, unit_node)
            else:
                raise NotImplementedError

        elif node.left.type.is_string and node.right.type.is_string:
            node.left = nodes.CoercionNode(node.left, object_)
            node.right = nodes.CoercionNode(node.right, object_)
            return nodes.CoercionNode(self.visit_BinOp(node), c_string_type)

        return node

    def _object_unaryop(self, node, api_name):
        return self.visit(
            function_util.external_call(self.context,
                                        self.llvm_module,
                                        api_name,
                                        args=[node.operand]))

    def _object_Invert(self, node):
        return self._object_unaryop(node, 'PyNumber_Invert')

    def _object_Not(self, node):
        callnode = function_util.external_call(self.function_cache,
                                               self.llvm_module,
                                               'PyObject_IsTrue',
                                               args=[node.operand])
        cmpnode = ast.Compare(callnode, [nodes.Eq()], [nodes.ConstNode(0)])
        return self.visit(nodes.IfExp(cmpnode,
                                      nodes.ObjectInjectNode(True),
                                      nodes.ObjectInjectNode(False)))

    def _object_UAdd(self, node):
        return self._object_unaryop(node, 'PyNumber_Positive')

    def _object_USub(self, node):
        return self._object_unaryop(node, 'PyNumber_Negative')

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if is_obj(node.type):
            op_name = type(node.op).__name__
            op_method = getattr(self, '_object_%s' % op_name, None)
            if op_method:
                node = op_method(node)
            else:
                raise error.NumbaError(
                    node, 'Unsupported unary operation for objects: %s' %
                    op_name)
        return node

    def visit_ConstNode(self, node):
        constant = node.pyval
        if node.type.is_known_value:
            node.type = object_ # TODO: Get rid of known_value

        if node.type.is_complex:
            real = nodes.ConstNode(constant.real, node.type.base_type)
            imag = nodes.ConstNode(constant.imag, node.type.base_type)
            node = nodes.ComplexNode(real, imag)

        elif node.type.is_numpy_datetime:
            datetime_str = nodes.ConstNode('', c_string_type)
            node = nodes.NumpyDateTimeNode(datetime_str)

        elif node.type.is_datetime:
            # JNB: not sure what to do here for datetime value
            timestamp = nodes.ConstNode(0, int64)
            units = nodes.ConstNode(0, int32)
            node = nodes.DateTimeNode(timestamp, units)

        elif node.type.is_timedelta:
            diff = nodes.ConstNode(0, int64)
            units = nodes.ConstNode(0, int32)
            node = nodes.TimeDeltaNode(diff, units)

        elif node.type.is_pointer and not node.type.is_string:
            addr_int = constnodes.get_pointer_address(constant, node.type)
            node = nodes.ptrfromint(addr_int, node.type)

        elif node.type.is_object and not nodes.is_null_constant(constant):
            node = nodes.ObjectInjectNode(constant, node.type)

        return node

    #------------------------------------------------------------------------
    # User nodes
    #------------------------------------------------------------------------

    def visit_UserNode(self, node):
        return node.specialize(self)
