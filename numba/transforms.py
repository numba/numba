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


import sys
import ast
import copy
import opcode
import types
import ctypes
import textwrap
import traceback
import __builtin__ as builtins

if __debug__:
    import pprint

import numba
from numba import *
from numba import error, closure
from .minivect import minierror, minitypes, codegen
from . import macros, utils, typesystem
from .symtab import Variable
from . import visitors, nodes, error, functions
from numba import stdio_util, function_util
from numba.typesystem import is_obj, promote_closest
from numba.utils import dump

import llvm.core
import numpy as np

logger = logging.getLogger(__name__)

from numba.external import pyapi

is_win32 = sys.platform == 'win32'

def filter_math_funcs(math_func_names):
    if is_win32:
        dll = ctypes.cdll.msvcrt
    else:
        dll = ctypes.CDLL(None)

    result_func_names = []
    for name in math_func_names:
        if getattr(dll, name, None) is not None:
            result_func_names.append(name)

    return result_func_names

class MathMixin(object):
    """
    Resolve calls to math functions.

    During type inference this produces MathNode nodes, and during
    final specialization it produces LLVMIntrinsicNode and MathCallNode
    nodes.
    """

    # sin(double), sinf(float), sinl(long double)
    libc_math_funcs = [
        'sin',
        'cos',
        'tan',
        'acos',
        'asin',
        'atan',
        'atan2',
        'sinh',
        'cosh',
        'tanh',
        'asinh',
        'acosh',
        'atanh',
        'log2',
        'log10',
        'fabs',
        'pow',
        'erfc',
        'ceil',
        'expm1',
        'rint',
        'log1p',
        'round',
    ]
    libc_math_funcs = filter_math_funcs(libc_math_funcs)

    def get_funcname(self, py_func):
        if py_func is np.abs:
            return 'fabs'
        elif py_func is np.round:
            return 'round'

        return py_func.__name__

    def _is_intrinsic(self, py_func):
        "Whether the math function is available as an llvm intrinsic"
        intrinsic_name = 'INTR_' + self.get_funcname(py_func).upper()
        is_intrinsic = hasattr(llvm.core, intrinsic_name)
        return is_intrinsic # and not is_win32

    def _is_math_function(self, func_args, py_func):
        if len(func_args) == 0 or len(func_args) > 1 or py_func is None:
            return False

        type = func_args[0].variable.type

        if type.is_array:
            type = type.dtype
            valid_type = type.is_float or type.is_int or type.is_complex
        else:
            valid_type = type.is_float or type.is_int

        is_intrinsic = self._is_intrinsic(py_func)

        math_name = self.get_funcname(py_func)
        is_math = math_name in self.libc_math_funcs
        if is_math and valid_type:
            math_name = self.math_suffix(math_name, type)
            is_math = filter_math_funcs([math_name])

        return valid_type and (is_intrinsic or is_math)

    def _resolve_intrinsic(self, args, py_func, signature):
        func_name = self.get_funcname(py_func).upper()
        return nodes.LLVMIntrinsicNode(signature, args, func_name=func_name)

    def math_suffix(self, name, type):
        if name == 'abs':
            name = 'fabs'

        if type.itemsize == 4:
            name += 'f' # sinf(float)
        elif type.itemsize == 16:
            name += 'l' # sinl(long double)
        return name

    def _resolve_libc_math(self, args, py_func, signature):
        arg_type = signature.args[0]
        name = self.math_suffix(self.get_funcname(py_func), arg_type)
        return nodes.MathCallNode(signature, args, llvm_func=None,
                                  py_func=py_func, name=name)

    def _resolve_math_call(self, call_node, py_func):
        "Resolve calls to math functions to llvm.log.f32() etc"
        # signature is a generic signature, build a correct one
        orig_type = type = call_node.args[0].variable.type

        if type.is_int:
            type = double
        elif type.is_array and type.dtype.is_int:
            type = type.copy(dtype=double)

        signature = minitypes.FunctionType(return_type=type, args=[type])
        result = nodes.MathNode(py_func, signature, call_node.args[0])
        return result

    def _binop_type(self, x, y):
        "Binary result type for math operations"
        x_type = x.variable.type
        y_type = y.variable.type
        dst_type = self.context.promote_types(x_type, y_type)
        type = dst_type
        if type.is_int:
            type = double

        signature = minitypes.FunctionType(return_type=type, args=[type, type])
        return dst_type, type, signature

    def pow(self, node, power, mod=None):
        name = 'pow'
        dst_type, pow_type, signature = self._binop_type(node, power)
        args = [node, power]
        if pow_type.is_float and mod is None:
            result = self._resolve_intrinsic(args, pow, signature)
        else:
            if mod is not None:
                args.append(mod)
            result = nodes.call_pyfunc(pow, args)

        return nodes.CoercionNode(result, dst_type)


class BuiltinResolverMixinBase(MathMixin):
    """
    Base class for mixins resolving calls to built-in functions.

    Methods called _resolve_<built-in name> are called to handle calls
    to the built-in of that name.
    """

    def _resolve_builtin_call(self, node, func):
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

    def _resolve_builtin_call_or_object(self, node, func):
        """
        Resolve an ast.Call() of a built-in function, or call the built-in
        through the object layer otherwise.
        """
        result = self._resolve_builtin_call(node, func)
        if result is None:
            result = nodes.call_pyfunc(func, node.args)

        return result

    def _expect_n_args(self, func, node, n):
        if not isinstance(n, tuple):
            n = (n,)

        if len(node.args) not in n:
            expected = " or ".join(map(str, n))
            raise error.NumbaError(
                node, "builtin %s expects %s arguments" % (func.__name__,
                                                           expected))

class LateBuiltinResolverMixin(BuiltinResolverMixinBase):
    """
    Perform final low-level transformations such as abs(value) -> fabs(value)
    """

    def _resolve_abs(self, func, node, argtype):
        self._expect_n_args(func, node, 1)

        is_math = self._is_math_function(node.args, abs)

        # TODO: generate efficient inline code
        if is_math and argtype.is_float:
            return self._resolve_math_call(node, abs)
        elif is_math and argtype.is_int:
            if argtype.signed:
                type = promote_closest(self.context, argtype, [long_, longlong])
                funcs = {long_: 'labs', longlong: 'llabs'}
                return function_util.external_call(
                                            self.context,
                                            self.llvm_module,
                                            funcs[type],
                                            args=[node.args[0]])
            else:
                # abs() on unsigned integral value
                return node.args[0]

        return None

    def _resolve_round(self, func, node, argtype):
        self._expect_n_args(func, node, (1, 2))
        if self._is_math_function(node.args, round):
            # round() always returns a float
            return self._resolve_math_call(node, round)

        return None

    def _resolve_pow(self, func, node, argtype):
        self._expect_n_args(func, node, (2, 3))
        return self.pow(*node.args)


def unpack_range_args(node):
    start, stop, step = (nodes.const(0, Py_ssize_t),
                         None,
                         nodes.const(1, Py_ssize_t))

    if len(node.args) == 0:
        raise error.NumbaError(node, "Expected at least one argument")
    elif len(node.args) == 1:
        stop, = node.args
    elif len(node.args) == 2:
        start, stop = node.args
    else:
        start, stop, step = node.args

    return [start, stop, step]

class TransformForIterable(visitors.NumbaTransformer):
    """
    This transforms for loops such as loops over 1D arrays:

            for value in my_array:
                ...

        into

            for i in my_array.shape[0]:
                value = my_array[i]
    """

    def visit_For(self, node):
        if node.iter.type.is_range:
            #
            ### Handle range iteration
            #
            self.generic_visit(node)

            temp = nodes.TempNode(node.target.type, 'target_temp')
            nsteps = nodes.TempNode(Py_ssize_t, 'nsteps')
            start, stop, step = unpack_range_args(node.iter)

            if isinstance(step, nodes.ConstNode):
                have_step = step.pyval != 1
            else:
                have_step = True

            start, stop, step = [nodes.CloneableNode(n)
                                     for n in (start, stop, step)]

            if have_step:
                compute_nsteps = """
                    $length = {{stop}} - {{start}}
                    {{nsteps}} = $length / {{step}}
                    if {{nsteps_load}} * {{step}} != $length: #$length % {{step}}:
                        # Test for truncation
                        {{nsteps}} = {{nsteps_load}} + 1
                    # print "nsteps", {{nsteps_load}}
                """
            else:
                compute_nsteps = "{{nsteps}} = {{stop}} - {{start}}"

            if node.orelse:
                else_clause = "else: {{else_body}}"
            else:
                else_clause = ""

            templ = textwrap.dedent("""
                %s
                {{temp}} = 0
                while {{temp_load}} < {{nsteps_load}}:
                    {{target}} = {{start}} + {{temp_load}} * {{step}}
                    {{body}}
                    {{temp}} = {{temp_load}} + 1
                %s
            """) % (textwrap.dedent(compute_nsteps), else_clause)

            # Leave the bodies empty, they are already analyzed
            body = ast.Suite(body=[])
            else_body = ast.Suite(body=[])

            # Substitute template and type infer
            result = self.run_template(
                templ, vars=dict(length=Py_ssize_t),
                start=start, stop=stop, step=step,
                nsteps=nsteps.store(), nsteps_load=nsteps.load(),
                temp=temp.store(), temp_load=temp.load(),
                target=node.target,
                body=body, else_body=else_body)

            # Patch the body and else clause
            body.body.extend(node.body)
            else_body.body.extend(node.orelse)

            # Patch cfg block of target variable to the 'body' block of the
            # while
            node.target.variable.block = node.if_block

            # Create the place to jump to for 'continue'
            while_node = result.body[-1]
            assert isinstance(while_node, ast.While)

            target_increment = while_node.body[-1]
            assert isinstance(target_increment, ast.Assign)
            #incr_block = nodes.LowLevelBasicBlockNode(node=target_increment,
            #                                          name='for_increment')
            node.incr_block.body = [target_increment]
            while_node.body[-1] = node.incr_block
            while_node.continue_block = node.incr_block

            # Patch the while with the For nodes cfg blocks
            attrs = dict(vars(node), **vars(while_node))
            while_node = nodes.build_while(**attrs)
            result.body[-1] = while_node

            return result

        elif node.iter.type.is_array and node.iter.type.ndim == 1:
            # Convert 1D array iteration to for-range and indexing
            logger.debug(ast.dump(node))

            orig_target = node.target
            orig_iter = node.iter

            # replace node.target with a temporary
            target_name = orig_target.id + '.idx'
            target_temp = nodes.TempNode(Py_ssize_t)
            node.target = target_temp.store()

            # replace node.iter
            call_func = ast.Name(id='range', ctx=ast.Load())
            call_func.type = typesystem.RangeType()
            call_func.variable = Variable(call_func.type)

            shape_index = ast.Index(nodes.ConstNode(0, typesystem.Py_ssize_t))
            shape_index.type = typesystem.npy_intp
            stop = ast.Subscript(value=nodes.ShapeAttributeNode(orig_iter),
                                 slice=shape_index,
                                 ctx=ast.Load())
            stop.type = typesystem.intp
            stop.variable = Variable(stop.type)
            call_args = [nodes.ConstNode(0, typesystem.Py_ssize_t),
                         nodes.CoercionNode(stop, typesystem.Py_ssize_t),
                         nodes.ConstNode(1, typesystem.Py_ssize_t),]

            node.iter = ast.Call(func=call_func, args=call_args)
            node.iter.type = call_func.type

            node.index = target_temp.load()
            # add assignment to new target variable at the start of the body
            index = ast.Index(value=node.index)
            index.type = target_temp.type
            subscript = ast.Subscript(value=orig_iter,
                                      slice=index, ctx=ast.Load())
            subscript.type = orig_iter.variable.type.dtype
            subscript.variable = Variable(subscript.type)
            coercion = nodes.CoercionNode(subscript, orig_target.type)
            assign = ast.Assign(targets=[orig_target], value=subscript)

            node.body = [assign] + node.body

            return self.visit(node)
        else:
            raise error.NumbaError("Unsupported for loop pattern")

class ResolveCoercions(visitors.NumbaTransformer):

    def visit_CoercionNode(self, node):
        if not isinstance(node, nodes.CoercionNode):
            # CoercionNode.__new__ returns the node to be coerced if it doesn't
            # need coercion
            return node

        node_type = node.node.type
        dst_type = node.dst_type
        if __debug__ and logger.getEffectiveLevel() < logging.DEBUG:
            logger.debug('coercion: %s --> %s\n%s',
                         node_type, dst_type, utils.pformat_ast(node))

        if self.nopython and is_obj(node_type):
            raise error.NumbaError(node, "Cannot coerce to or from object in "
                                         "nopython context")

        if is_obj(node.dst_type) and not is_obj(node_type):
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
            result = self.visit(ast.Compare(node.node, [ast.NotEq()],
                                            [nodes.const(0, node_type)]))
        elif node_type.is_c_string and dst_type.is_numeric:
            # TODO: int <-> string conversions are explicit, this should not
            # TODO: be a coercion
            if self.nopython:
                node = nodes.CoercionNode(
                    function_util.external_call(
                                self.context,
                                self.llvm_module,
                                ('atol' if dst_type.is_int else 'atof'),
                                args=[node.node]),
                    dst_type, name=node.name,)
            else:
                if dst_type.is_int:
                    cvtobj = function_util.external_call(
                                              self.context,
                                              self.llvm_module,
                                              'PyInt_FromString',
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
        else:
            self.generic_visit(node)
#            if not node.node.type == node_type:
#                result = self.visit(node)
            if dst_type == node.node.type:
                result = node.node
            else:
                result = node

        if __debug__ and logger.getEffectiveLevel() < logging.DEBUG:
            logger.debug('result = %s', utils.pformat_ast(result))

        return result

    def _get_int_conversion_func(self, type, funcs_dict):
        type = self.context.promote_types(type, long_)
        if type in funcs_dict:
            return funcs_dict[type]

        if type.itemsize == long_.itemsize:
            types = [ulong, long_]
        else:
            types = [ulonglong, longlong]

        return self._get_int_conversion_func(types[type.signed], funcs_dict)

    def visit_CoerceToObject(self, node):
        new_node = node

        node_type = node.node.type
        if node_type.is_numeric:
            cls = None
            args = node.node,
            if node_type.is_int:
                cls = self._get_int_conversion_func(node_type,
                                                    pyapi._from_long)
            elif node_type.is_float:
                cls = pyapi.PyFloat_FromDouble
            elif node_type.is_complex:
                cls = pyapi.PyComplex_FromDoubles
                complex_value = nodes.CloneableNode(node.node)
                args = [
                    nodes.ComplexAttributeNode(complex_value, "real"),
                    nodes.ComplexAttributeNode(complex_value.clone, "imag")
                ]
            else:
                raise error.NumbaError(
                    node, "Don't know how to coerce type %r to PyObject" %
                    node_type)

            if cls:
                new_node = function_util.external_call(self.context,
                                                       self.llvm_module,
                                                       cls.__name__,
                                                       args=args)
        elif node_type.is_pointer and not node_type.is_string():
            # Create ctypes pointer object
            ctypes_pointer_type = node_type.to_ctypes()
            args = [nodes.CoercionNode(node.node, int64),
                    nodes.ObjectInjectNode(ctypes_pointer_type, object_)]
            new_node = nodes.call_pyfunc(ctypes.cast, args)
        elif node_type.is_bool:
            new_node = function_util.external_call(self.context,
                                                   self.llvm_module,
                                                   "PyBool_FromLong",
                                                   args=[node.node])

        self.generic_visit(new_node)
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
                cls = self._get_int_conversion_func(node_type,
                                                    pyapi._as_long)
                if not node_type.signed or node_type == Py_ssize_t:
                    # PyLong_AsLong calls __int__, but
                    # PyLong_AsUnsignedLong doesn't...
                    node.node = nodes.call_pyfunc(long, [node.node])
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
        elif node_type.is_pointer:
            raise error.NumbaError(node, "Obtaining pointers from objects "
                                         "is not yet supported")
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

def badval(type):
    if type.is_object or type.is_array:
        value = nodes.NULL_obj
        if type != object_:
            value = value.coerce(type)
    elif type.is_void:
        value = None
    elif type.is_float:
        value = nodes.ConstNode(float('nan'), type=type)
    elif type.is_int or type.is_complex:
        # TODO: adjust for type.itemsize
        bad = 0xbadbad # This pattern is hard to detect in llvm code
        bad = 123456789
        value = nodes.ConstNode(bad, type=type)
    else:
        value = nodes.BadValue(type)

    return value

class LateSpecializer(closure.ClosureCompilingMixin, ResolveCoercions,
                      LateBuiltinResolverMixin, visitors.NoPythonContextMixin):

    def visit_FunctionDef(self, node):
        self.handle_phis()

        node.decorator_list = self.visitlist(node.decorator_list)
        node.body = self.visitlist(node.body)

        ret_type = self.func_signature.return_type
        if ret_type.is_object or ret_type.is_array:
            # This will require some increfs, but allow it if people
            # use 'with python' later on. If 'with python' isn't used, a
            # return will issue the error
            #if self.nopython:
            #    raise error.NumbaError(
            #            node, "Function cannot return object in "
            #                  "nopython context")
            pass

        value = badval(ret_type)
        if value is not None:
            value = nodes.CoercionNode(value, dst_type=ret_type).cloneable

        error_return = ast.Return(value=value)
        if self.nopython and is_obj(self.func_signature.return_type):
            error_return = nodes.WithPythonNode(body=[error_return])

        node.error_return = error_return
        return node

    def handle_phi(self, node):
        """
        Handle phi nodes:

            1) Handle incoming variables which are not initialized. Set
               incoming_variable.uninitialized_value to a constant 'bad'
               value (e.g. 0xbad for integers, NaN for floats, NULL for
               objects)

            2) Handle incoming variables which need promotions. An incoming
               variable needs a promotion if it has a different type than
               the the phi. The promotion happens in each ancestor block that
               defines the variable which reaches us.

               Promotions are set separately in the symbol table, since the
               ancestor may not be our immediate parent, we cannot introduce
               a rename and look up the latest version since there may be
               multiple different promotions. So during codegen, we first
               check whether incoming_type == phi_type, and otherwise we
               look up the promotion in the parent block or an ancestor.
        """
        for parent_block, incoming_var in node.find_incoming():
            if incoming_var.type.is_uninitialized:
                incoming_type = incoming_var.type.base_type or node.type
                bad = badval(incoming_type)
                incoming_var.type.base_type = incoming_type
                incoming_var.uninitialized_value = self.visit(bad)
                # print incoming_var

            elif not incoming_var.type == node.type:
                # Create promotions for variables with phi nodes in successor
                # blocks.
                incoming_symtab = incoming_var.block.symtab
                if (incoming_var, node.type) not in node.block.promotions:
                    # Make sure we only coerce once for each destination type and
                    # each variable
                    incoming_var.block.promotions.add((incoming_var, node.type))

                    # Create promotion node
                    name_node = nodes.Name(id=incoming_var.renamed_name,
                                           ctx=ast.Load())
                    name_node.variable = incoming_var
                    name_node.type = incoming_var.type
                    coercion = self.visit(name_node.coerce(node.type))
                    promotion = nodes.PromotionNode(node=coercion)

                    # Add promotion node to block body
                    incoming_var.block.body.append(promotion)
                    promotion.variable.block = incoming_var.block

                    # Update symtab
                    incoming_symtab.promotions[incoming_var.name,
                                               node.type] = promotion
                else:
                    promotion = incoming_symtab.lookup_promotion(
                                     incoming_var.name, node.type)

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

        stdin, stdout, stderr = stdio_util.get_stdio_streams()
        stdout = stdio_util.get_stream_as_node(stdout)

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
            dst_type = c_string_type
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
        n = nodes.ConstNode(len(node.elts), minitypes.Py_ssize_t)
        args = [n] + objs
        new_node = nodes.NativeCallNode(sig, args, lfunc, name='tuple')
        new_node.type = typesystem.TupleType(size=len(node.elts))
        return nodes.ObjectTempNode(new_node)

    def visit_List(self, node):
        self.check_context(node)
        self.generic_visit(node)
        return nodes.ObjectTempNode(node)

    def visit_Dict(self, node):
        self.check_context(node)
        self.generic_visit(node)
        return nodes.ObjectTempNode(node)

    def visit_Call(self, node):
        func_type = node.func.type

        if func_type.is_builtin:
            result = self._resolve_builtin_call_or_object(node, func_type.func)
            result =  self.visit(result)
            return result

        self.generic_visit(node)
        return node

    def visit_NativeCallNode(self, node):
        if is_obj(node.signature.return_type):
            if self.nopython:
                raise error.NumbaError(
                    node, "Cannot call function returning object in "
                          "nopython context")

            self.generic_visit(node)
            return nodes.ObjectTempNode(node)
        elif node.badval is not None:
            result = node.cloneable
            body = nodes.CheckErrorNode(
                        result, node.badval, node.goodval,
                        node.exc_type, node.exc_msg, node.exc_args)
            node = nodes.ExpressionNode(stmts=[body],
                                        expr=result.clone)
            return self.visit(node)
        else:
            self.generic_visit(node)
            return node

    def visit_NativeFunctionCallNode(self, node):
        if node.signature.is_bound_method:
            assert isinstance(node.function, nodes.ExtensionMethod)
            return self.visit_ExtensionMethod(node.function, node)

        return self.visit_NativeCallNode(node)

    def visit_ObjectCallNode(self, node):
        # self.generic_visit(node)
        assert node.function

        if self.nopython:
            raise error.NumbaError(node, "Cannot use object call in "
                                         "nopython context")

        node.function = self.visit(node.function)
        node.args_tuple = self.visit(node.args_tuple)
        node.kwargs_dict = self.visit(node.kwargs_dict)
        return nodes.ObjectTempNode(node)

    def visit_MathNode(self, math_node):
        "Translate a nodes.MathNode to an intrinsic or libc math call"
        if math_node.type.is_array:
            assert math_node.py_func is not None
            result = nodes.call_pyfunc(math_node.py_func, [math_node.arg])
            return self.visit(result.coerce(math_node.type))

        args = [math_node.arg], math_node.py_func, math_node.signature
        if self._is_intrinsic(math_node.py_func):
            result = self._resolve_intrinsic(*args)
        else:
            result = self._resolve_libc_math(*args)

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
        if (self.nopython and node.value.type.is_c_string and
                node.type.is_c_string):
            return self.visit(self._c_string_slice(node))

        # logging.debug(ast.dump(node))
        # TODO: do this in the respective cases below when needed
        self.generic_visit(node)

        node_type = node.value.type
        if node_type.is_object or (node_type.is_array and
                                   node.slice.type.is_object):
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
        elif (node.value.type.is_array and not node.type.is_array and
                  node.slice.type.is_int):
            # Array index with integer indices
            node = nodes.DataPointerNode(node.value, node.slice, node.ctx)
        elif node.value.type.is_c_string and node.type.is_c_string:
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
        if self.nopython:
            raise error.NumbaError(
                    node, "Cannot access Python attribute in nopython context")

        if node.value.type.is_complex:
            value = self.visit(node.value)
            return nodes.ComplexAttributeNode(value, node.attr)
        elif node.type.is_numpy_attribute:
            return nodes.ObjectInjectNode(node.type.value)
        elif node.type.is_numpy_dtype:
            dtype_type = node.type.resolve()
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

    def visit_ExtTypeAttribute(self, node):
        """
        Resolve an extension attribute:

            ((attributes_struct *)
                 (((char *) obj) + attributes_offset))->attribute
        """
        ext_type = node.value.type
        offset = ext_type.attr_offset
        type = ext_type.attribute_struct

        if isinstance(node.ctx, ast.Load):
            value_type = type.ref()         # Load result
        else:
            value_type = type.pointer()     # Use pointer for storage

        struct_pointer = nodes.value_at_offset(node.value, offset,
                                               value_type)
        result = nodes.StructAttribute(struct_pointer, node.attr,
                                       node.ctx, type.ref())

        return self.visit(result)

    def visit_ExtensionMethod(self, node, call_node=None):
        """
        Resolve an extension method:

            typedef {
                double (*method1)(double);
                ...
            } vtab_struct;

            vtab_struct *vtab = *(vtab_struct **) (((char *) obj) + vtab_offset)
            void *method = vtab[index]
        """
        if call_node is None:
            raise error.NumbaError(node, "Referenced extension method '%s' "
                                         "must be called" % node.attr)

        # Make the object we call the method on clone-able
        node.value = nodes.CloneableNode(self.visit(node.value))

        ext_type = node.value.type
        offset = ext_type.vtab_offset
        struct_type = ext_type.vtab_type.ref()

        struct_pointer_pointer = nodes.value_at_offset(node.value, offset,
                                                       struct_type.pointer())
        struct_pointer = nodes.DereferenceNode(struct_pointer_pointer)

        vmethod = nodes.StructAttribute(struct_pointer, node.attr,
                                        ast.Load(), struct_type)

        # Visit argument list for call
        args = self.visitlist(call_node.args)

        # Insert first argument 'self' in args list
        args.insert(0, nodes.CloneNode(node.value))
        result = nodes.NativeFunctionCallNode(node.type, vmethod, args)
        result.signature.is_bound_method = False

        return self.visit(result)

    def visit_ArrayNewNode(self, node):
        if self.nopython:
            # Give the codegen (subclass) a chance to handle this
            self.generic_visit(node)
            return node

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

        array = nodes.PyArray_NewFromDescr(args).cloneable
        body = [incref_descr, incref_base, array, setbase]

        if node.base is not None:
            body.append(nodes.PyArray_SetBaseObject([array.clone, base.clone]))

        # TODO: PyArray_UpdateFlags()
        result = nodes.ExpressionNode(filter(None, body), array.clone)
        return self.visit(result)

    def visit_ArrayNewEmptyNode(self, node):
        ndim = nodes.const(node.type.ndim, int_)
        dtype = nodes.const(node.type.dtype.get_dtype(), object_)
        is_fortran = nodes.const(node.is_fortran, int_)
        result = nodes.PyArray_Empty([ndim, node.shape, dtype, is_fortran])
        return self.visit(result)

    def _raise_exception(self, body, node):
        if node.exc_type:
            assert node.exc_msg

            if node.exc_args:
                args = [node.exc_type, node.exc_msg, node.exc_args]
                raise_node = function_util.external_call(self.context,
                                                         self.llvm_module,
                                                         'PyErr_Format',
                                                         args=args)
            else:
                args = [node.exc_type, node.exc_msg]
                raise_node = function_util.external_call(self.context,
                                                         self.llvm_module,
                                                        'PyErr_SetString',
                                                         args=args)

            body.append(raise_node)

    def _trap(self, body, node):
        if node.exc_msg and node.print_on_trap:
            pos = error.format_pos(node)
            if node.exception_type:
                exc_type = '%s: ' % node.exception_type.__name__
            else:
                exc_type = ''

            msg = '%s%s%%s' % (exc_type, pos)
            format = nodes.const(msg, c_string_type)
            print_msg = function_util.external_call(self.context,
                                                    self.llvm_module,
                                                    'printf',
                                                    args=[format,
                                                          node.exc_msg])
            body.append(print_msg)

        trap = nodes.LLVMIntrinsicNode(signature=void(), args=[],
                                       func_name='TRAP')
        body.append(trap)

    def visit_RaiseNode(self, node):
        body = []
        if self.nopython:
            self._trap(body, node)
        else:
            self._raise_exception(body, node)

        body.append(nodes.PropagateNode())
        return ast.Suite(body=body)

    def visit_CheckErrorNode(self, node):
        if node.badval is not None:
            badval = node.badval
            eq = ast.Eq()
        else:
            assert node.goodval is not None
            badval = node.goodval
            eq = ast.NotEq()

        test = ast.Compare(left=node.return_value, ops=[eq],
                           comparators=[badval])
        test.right = badval

        check = nodes.build_if(test=test, body=[node.raise_node], orelse=[])
        return self.visit(check)

    def visit_Name(self, node):
        if node.type.is_builtin and not node.variable.is_local:
            obj = getattr(builtins, node.name)
            return nodes.ObjectInjectNode(obj, node.type)

        if (is_obj(node.type) and isinstance(node.ctx, ast.Load) and
                getattr(node, 'cf_maybe_null', False)):
            # Check for unbound objects and raise UnboundLocalError if so
            value = nodes.LLVMValueRefNode(Py_uintptr_t, None)
            node.loaded_name = value

            exc_msg = node.variable.name
            if hasattr(node, 'lineno'):
               exc_msg = '%s%s' % (error.format_pos(node), exc_msg)

            check_unbound = nodes.CheckErrorNode(
                    value, badval=nodes.const(0, Py_uintptr_t),
                    exc_type=UnboundLocalError,
                    exc_msg=exc_msg)
            node.check_unbound = self.visit(check_unbound)

        return super(LateSpecializer, self).visit_Name(node)

    def visit_Return(self, node):
        return_type = self.func_signature.return_type
        if node.value is not None:
            node.value = self.visit(nodes.CoercionNode(node.value, return_type))
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        return node

    def visit_Compare(self, node):
        if node.left.type.is_pointer and node.comparators[0].type.is_pointer:
            node.left = nodes.CoercionNode(node.left, Py_uintptr_t)
            node.comparators = [nodes.CoercionNode(node.comparators[0],
                                                   Py_uintptr_t)]

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
        cmpnode = nodes.Compare(callnode, [nodes.Eq()], [nodes.ConstNode(0)])
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

    #------------------------------------------------------------------------
    # User nodes
    #------------------------------------------------------------------------

    def visit_UserNode(self, node):
        return node.specialize(self)
