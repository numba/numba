import ast
import inspect
import types

import numba
from numba import *
from numba.minivect import minitypes
from numba import typesystem, symtab

import numpy.random
import numpy as np

import logging

debug = False
#debug = True

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
if debug:
    logger.setLevel(logging.DEBUG)


class ModuleTypeInfererRegistry(object):
    "Builds the module type inferers for the modules we can handle"

    is_registered = False

    def register(self, context):
        if not self.is_registered:
            NumbaModuleInferer(context).register()
            NumpyModuleInferer(context).register()
            self.is_registered = True

module_registry = ModuleTypeInfererRegistry()

def module_attribute_type(obj):
    """
    See if the object is registered to any module which might handle
    type inference on the object.
    """
    try:
        is_module_attribute_type = obj in ModuleTypeInferer.member2inferer
    except TypeError:
        pass # unhashable object
    else:
        if is_module_attribute_type:
            inferer = ModuleTypeInferer.member2inferer[obj]
            module = inferer.member_modules[obj]
            attr = inferer.members[obj]
            return typesystem.ModuleAttributeType(module=module, attr=attr)

    return None

def parse_args(call_node, arg_names):
    "Parse positional and keyword arguments"
    result = dict.fromkeys(arg_names)

    # parse positional arguments
    i = 0
    for i, (arg_name, arg) in enumerate(zip(arg_names, call_node.args)):
        result[arg_name] = arg

    arg_names = arg_names[i:]
    if arg_names:
        # parse keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg in result:
                result[keyword.arg] = keyword.value

    return result


class ModuleTypeInferer(object):
    """
    Represents functionality that has module-specific knowledge.

    Attributes and names with object from modules listed in the modules list
    create a ModuleAttributeType, whereas calling them will resolve to methods
    on this object.
    """

    # Modules this module type inferer can handle
    modules = None

    registered_inferers = []
    member2inferer = {}

    def __init__(self, context, modules=None):
        self.context = context
        self.modules = modules or self.modules

        # NOTE: __name__ may be unreliable, we use the name exposed by the
        # module!
        # members: { np.zeros: 'zeros' }
        # member_modules: { np.zeros: numpy.core.multiarray }
        self.members, self.member_modules = self.build_table()

    def build_table(self):
        """
        Collects the set of attributes which are recognized by this module
        type inferer. This may be needed for objects without __module__
        that are used as globals or are otherwise passed into numba functions,
        e.g.:

            from numpy import zeros

            @autojit
            def func():
                A = zeros(...)
        """
        all_members = {}
        all_member_modules = {}

        def predicate(obj):
            #try:
            #    return doctest_support.from_module(module, obj)
            #except ValueError:
            #    return True
            have_module = getattr(obj, "__module__", None)
            if have_module:
                return obj.__module__.startswith(module.__name__)

            return True

        for module in self.modules:
            # Members defined in this module
            members = inspect.getmembers(module, predicate)
            for name, obj in members:
                if self.is_member(name, obj):
                    try:
                        all_members[obj] = name
                        all_member_modules[obj] = module
                    except TypeError:
                        # Unhashable
                        # print name
                        pass

        return all_members, all_member_modules

    def is_member(self, name, obj):
        valid_types = (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.ClassType,
            type,
            np.ufunc,
        )
        return isinstance(obj, valid_types)

    def dispatch_on_name(self, call_node, func_type):
        """
        Dispatch a call of an attribute by its __name__ attribute.

        For instance, a method

            def empty(shape, dtype, order):
                ...

        would be called with those arguments. Parameters not present as
        arguments in user code are None.

        Returns the result type, or None
        """
        obj = func_type.value
        if obj not in self.members:
            return

        name = self.members[obj]
        method = getattr(self, name, None)
        if method is None:
            return

        argnames = inspect.getargspec(method.im_func).args
        assert argnames.pop(0) == "self" # remove 'self' argument
        method_kwargs = parse_args(call_node, argnames)
        return method(**method_kwargs)

    def resolve_call(self, call_node, obj_call_node, func_type):
        """
        Call to an attribute of this module.

            func_type: ModuleAttributeType
                |__________> module: Python module
                |__________> attr: Attribute name
                |__________> value: Attribute value

        The default is to dispatch on the name of the member.
        """
        result = self.dispatch_on_name(call_node, func_type)

        if result is not None and not isinstance(result, ast.AST):
            assert isinstance(result, minitypes.Type)
            type = result
            result = obj_call_node
            result.variable = symtab.Variable(type)

        return result

    def register(self):
        self.member2inferer.update(dict.fromkeys(self.members, self))

    def get_module(self, member):
        self.member_modules[member]

class NumbaModuleInferer(ModuleTypeInferer):
    """
    Infer types for the Numba module.
    """

    modules = [numba]

    def typeof(self, expr):
        from numba import nodes

        obj = expr.variable.type
        type = typesystem.CastType(obj)
        return nodes.const(obj, type)

class NumpyModuleInferer(ModuleTypeInferer):
    """
    Infer types for NumPy functionality. This includes:

        1) Figuring out dtypes

            e.g. np.double     -> double
                 np.dtype('d') -> double

        2) Function calls such as np.empty/np.empty_like/np.arange/etc
    """

    modules = [numpy, numpy.random]

    def _resolve_attribute_dtype(self, dtype, default=None):
        "Resolve the type for numpy dtype attributes"
        if dtype.is_numpy_dtype:
            return dtype

        if dtype.is_numpy_attribute:
            numpy_attr = getattr(dtype.module, dtype.attr, None)
            if isinstance(numpy_attr, numpy.dtype):
                return typesystem.NumpyDtypeType(dtype=numpy_attr)
            elif issubclass(numpy_attr, numpy.generic):
                return typesystem.NumpyDtypeType(dtype=numpy.dtype(numpy_attr))

    def get_dtype(self, dtype_arg, default_dtype=None):
        "Get the dtype keyword argument from a call to a numpy attribute."
        if dtype_arg is None:
            if default_dtype is None:
                return None
            dtype = typesystem.NumpyDtypeType(dtype=numpy.dtype(default_dtype))
            return dtype
        else:
            return self._resolve_attribute_dtype(dtype_arg.variable.type)

    #------------------------------------------------------------------------
    # Resolution of NumPy calls
    #------------------------------------------------------------------------

    def dtype(self, obj, align):
        "Parse np.dtype(...) calls"
        if obj is None:
            return

        return self.get_dtype(obj)

    def empty_like(self, a, dtype, order):
        "Parse the result type for np.empty_like calls"
        if a is None:
            return

        type = a.variable.type
        if type.is_array:
            if dtype:
                dtype = self.get_dtype(dtype)
                if dtype is None:
                    return type
                dtype = dtype.resolve()
            else:
                dtype = type.dtype

            return minitypes.ArrayType(dtype, type.ndim)

    zeros_like = ones_like = empty_like

    def empty(self, shape, dtype, order):
        if shape is None:
            return None

        dtype = self.get_dtype(dtype, np.float64)
        shape_type = shape.variable.type

        if shape_type.is_int:
            ndim = 1
        elif shape_type.is_tuple or shape_type.is_list:
            ndim = shape_type.size
        else:
            return None

        return minitypes.ArrayType(dtype.resolve(), ndim)

    zeros = ones = empty

    def arange(self, start, stop, step, dtype):
        "Resolve a call to np.arange()"
        # NOTE: dtype must be passed as a keyword argument, or as the fourth
        # parameter
        dtype = self.get_dtype(dtype, numpy.int64)
        if dtype is not None:
            # return a 1D array type of the given dtype
            return dtype.resolve()[:]

    def dot(self, a, b, out):
        if out is not None:
            return out.variable.type

        lhs_type = promote_to_array(a.variable.type)
        rhs_type = promote_to_array(b.variable.type)

        dtype = self.context.promote_types(lhs_type.dtype, rhs_type.dtype)
        dst_ndim = lhs_type.ndim + rhs_type.ndim - 2

        result_type = minitypes.ArrayType(dtype, dst_ndim, is_c_contig=True)
        return result_type


def promote_to_array(dtype):
    if not dtype.is_array:
        dtype = minitypes.ArrayType(dtype, 0)
    return dtype