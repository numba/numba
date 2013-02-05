import __builtin__ as builtins

from numba import *
from numba import nodes
from numba import error
from numba import function_util
from numba.symtab import Variable
from numba import typesystem
from numba.typesystem import is_obj, promote_closest

from numba.type_inference.module_type_inference import (register,
                                                        register_inferer,
                                                        register_unbound)


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


class BuiltinResolverMixin(BuiltinResolverMixinBase):
    """
    Resolve builtin calls for type inference. Only applies high-level
    transformations such as type coercions. A subsequent pass in
    LateSpecializer performs low-level transformations.
    """

    def _resolve_range(self, func, node, argtype):
        node.variable = Variable(typesystem.RangeType())
        args = self.visitlist(node.args)
        node.args = nodes.CoercionNode.coerce(args, dst_type=Py_ssize_t)
        return node

    _resolve_xrange = _resolve_range

    def _resolve_len(self, func, node, argtype):
        # Simplify len(array) to ndarray.shape[0]
        self._expect_n_args(func, node, 1)
        if argtype.is_array:
            shape_attr = nodes.ArrayAttributeNode('shape', node.args[0])
            new_node = nodes.index(shape_attr, 0)
            return self.visit(new_node)

        return None

    dst_types = {
        int: numba.int_,
        float: numba.double,
        complex: numba.complex128
    }

    def _resolve_int(self, func, node, argtype):
        # Resolve int(x) and float(x) to an equivalent cast
        self._expect_n_args(func, node, (0, 1, 2))
        dst_type = self.dst_types[func]

        if len(node.args) == 0:
            return nodes.ConstNode(func(0), dst_type)
        elif len(node.args) == 1:
            return nodes.CoercionNode(node.args[0], dst_type=dst_type)
        else:
            # XXX Moved the unary version to the late specializer,
            # what about the 2-ary version?
            arg1, arg2 = node.args
            if arg1.variable.type.is_c_string:
                assert dst_type.is_int
                return nodes.CoercionNode(
                    nodes.ObjectTempNode(
                        function_util.external_call(
                            self.context,
                            self.llvm_module,
                            'PyInt_FromString',
                            args=[arg1, nodes.NULL, arg2])),
                    dst_type=dst_type)
            return None

    _resolve_float = _resolve_int

    def _resolve_complex(self, func, node, argtype):
        if len(node.args) == 2:
            args = nodes.CoercionNode.coerce(node.args, double)
            result = nodes.ComplexNode(real=args[0], imag=args[1])
        else:
            result = self._resolve_int(func, node, argtype)

        return result

    def _resolve_abs(self, func, node, argtype):
        self._expect_n_args(func, node, 1)

        # Result type of the substitution during late
        # specialization
        result_type = object_

        # What we actually get back regardless of implementation,
        # e.g. abs(complex) goes throught the object layer, but we know the result
        # will be a double
        dst_type = argtype

        is_math = self._is_math_function(node.args, abs)

        if argtype.is_complex:
            dst_type = double
        elif is_math and argtype.is_int and argtype.signed:
            # Use of labs or llabs returns long_ and longlong respectively
            result_type = promote_closest(self.context, argtype,
                                          [long_, longlong])
        elif is_math and (argtype.is_float or argtype.is_int):
            result_type = argtype

        node.variable = Variable(result_type)
        return nodes.CoercionNode(node, dst_type)

    def _resolve_pow(self, func, node, argtype):
        self._expect_n_args(func, node, (2, 3))
        return self.pow(*node.args)

    def _resolve_round(self, func, node, argtype):
        self._expect_n_args(func, node, (1, 2))
        is_math = self._is_math_function(node.args, round)
        if len(node.args) == 1 and argtype.is_int:
            # round(myint) -> myint
            return nodes.CoercionNode(node.args[0], double)

        if (argtype.is_float or argtype.is_int) and is_math:
            dst_type = argtype
        else:
            dst_type = object_
            node.args[0] = nodes.CoercionNode(node.args[0], object_)

        node.variable = Variable(dst_type)
        return nodes.CoercionNode(node, double)

    def _resolve_globals(self, func, node, argtype):
        self._expect_n_args(func, node, 0)
        return nodes.ObjectInjectNode(self.func.func_globals)

    def _resolve_locals(self, func, node, argtype):
        self._expect_n_args(func, node, 0)
        raise error.NumbaError("locals() is not supported in numba functions")

#----------------------------------------------------------------------------
#
#----------------------------------------------------------------------------

def _expect_n_args(func, node, n):
    if not isinstance(n, tuple):
        n = (n,)

    if len(node.args) not in n:
        expected = " or ".join(map(str, n))
        raise error.NumbaError(
            node, "builtin %s expects %s arguments" % (func.__name__,
                                                       expected))

def register_builtin(nargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(context, *args):
            _expect_n_args()
            return func(context, *args)

        wrapper.__name__ = wrapper.__name__.strip("_")
        register(builtins)(wrapper, pass_in_types=False)
        return func # wrapper

    return decorator

@register_builtin((1, 2, 3))
def range_(context, node, start, stop, step):
    node.variable = Variable(typesystem.RangeType())
    args = self.visitlist(node.args)
    node.args = nodes.CoercionNode.coerce(args, dst_type=Py_ssize_t)
    return node

@register_builtin((1, 2, 3))
def xrange_(context, node, start, stop, step):
    return range_(context, node, start, stop, step)

@register_builtin(1)
def len_(context, node, obj):
    # Simplify len(array) to ndarray.shape[0]
    argtype = get_type(obj)
    if argtype.is_array:
        shape_attr = nodes.ArrayAttributeNode('shape', node.args[0])
        new_node = nodes.index(shape_attr, 0)
        return new_node

    return Py_ssize_t

def cast(node, dst_type):
    if len(node.args) == 0:
        return nodes.ConstNode(0, dst_type)
    else:
        return nodes.CoercionNode(node.args[0], dst_type=dst_type)

@register_builtin((0, 1, 2))
def int_(context, node, x, base):
    # Resolve int(x) and float(x) to an equivalent cast
    dst_type = numba.int_

    if len(node.args) == 2:
        # XXX Moved the unary version to the late specializer,
        # what about the 2-ary version?
        arg1, arg2 = node.args
        if arg1.variable.type.is_c_string:
            assert dst_type.is_int
            return nodes.CoercionNode(
                nodes.ObjectTempNode(
                    function_util.external_call(
                        self.context,
                        self.llvm_module,
                        'PyInt_FromString',
                        args=[arg1, nodes.NULL, arg2])),
                dst_type=dst_type)

        return None

    else:
        return cast(node, int_)

@register_builtin((0, 1))
def float_(context, node, x):
    return cast(node, double)

@register_builtin((0, 1, 2))
def complex_(context, node, a, b):
    if len(node.args) == 2:
        args = nodes.CoercionNode.coerce(node.args, double)
        return nodes.ComplexNode(real=args[0], imag=args[1])
    else:
        return cast(node, complex128)

@register_builtin(1)
def _resolve_abs(context, node, x):
    # Result type of the substitution during late
    # specialization
    result_type = object_
    argtype = get_type(x)

    # What we actually get back regardless of implementation,
    # e.g. abs(complex) goes throught the object layer, but we know the result
    # will be a double
    dst_type = argtype

    is_math = _is_math_function(node.args, abs)

    if argtype.is_complex:
        dst_type = double
    elif is_math and argtype.is_int and argtype.signed:
        # Use of labs or llabs returns long_ and longlong respectively
        result_type = promote_closest(context, argtype,
                                      [long_, longlong])
    elif is_math and (argtype.is_float or argtype.is_int):
        result_type = argtype

    node.variable = Variable(result_type)
    return nodes.CoercionNode(node, dst_type)

@register_builtin((2, 3))
def _resolve_pow(context, node, base, exponent):
    # TODO:
    return self.pow(*node.args)

@register_builtin((1, 2))
def _resolve_round(context, node, number, ndigits):
    is_math = _is_math_function(node.args, round)
    argtype = get_type(number)

    if len(node.args) == 1 and argtype.is_int:
        # round(myint) -> myint
        return nodes.CoercionNode(node.args[0], double)

    if (argtype.is_float or argtype.is_int) and is_math:
        dst_type = argtype
    else:
        dst_type = object_
        node.args[0] = nodes.CoercionNode(node.args[0], object_)

    node.variable = Variable(dst_type)
    return nodes.CoercionNode(node, double)

@register_builtin(0)
def _resolve_globals(context, node):
    return typesystem.dict_
    # return nodes.ObjectInjectNode(func.func_globals)

@register_builtin(0)
def locals(context, node):
    raise error.NumbaError("locals() is not supported in numba functions")
