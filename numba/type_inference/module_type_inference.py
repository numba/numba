"""
Support for type functions for external code.

See modules/numpy*.py for type inference for NumPy.
"""

import ast
import inspect
import types

import numba
from numba import *
from numba.minivect import minitypes
from numba import typesystem, symtab, error, nodes
from numba.typesystem import get_type, typeset

import numpy.random
import numpy as np

import logging

debug = False
#debug = True

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
if debug:
    logger.setLevel(logging.DEBUG)

#----------------------------------------------------------------------------
# Exceptions
#----------------------------------------------------------------------------

class ValueAlreadyRegistered(error.NumbaError):
    """
    Raised when a type inferer is registered multiple times for the same
    value.
    """

class UnmatchedTypeError(error.NumbaError):
    """
    Raised when no matching specialization is found for a registered
    signature (`register_callable`).
    """

#----------------------------------------------------------------------------
# Global Registry for Type Functions
#----------------------------------------------------------------------------

class ModuleTypeInfererRegistry(object):
    "Builds the module type inferers for the modules we can handle"

    def __init__(self):
        super(ModuleTypeInfererRegistry, self).__init__()

        # function calls: (np.add)
        #     { value                               : (typefunc, pass_in_types) }
        # unbound methods: (np.add.reduce)
        #     { (value, unbound_dotted_path, False) : (typefunc, pass_in_types) }
        # bound methods: (obj.method where type(obj) is registered)
        #     { (type, bound_dotted_path, True)     : (typefunc, pass_in_types) }
        self.value_to_inferer = {}

        # { value : (module, 'attribute') }  (e.g. {np.add : (np, 'add')})
        self.value_to_module = {}

    def is_registered(self, value, func_type=None):
        try:
            hash(value)
        except TypeError:
            return False # unhashable object
        else:
            return value in self.value_to_inferer

    def register_inferer(self, module, attr, inferer, pass_in_types=True):
        """
        Register an type function (a type inferer) for a known function value.

            E.g. np.add() can be mapped as follows:

                module=np, attr='add', inferrer=my_inferer
        """
        value = getattr(module, attr)
        if self.is_registered(value):
            raise ValueAlreadyRegistered((value, module, inferer))

        self.value_to_module[value] = (module, attr)
        self.value_to_inferer[value] = (inferer, pass_in_types)

    def register_value(self, value, inferer, pass_in_types=True):
        self.value_to_inferer[value] = (inferer, pass_in_types)

    def register_unbound_method(self, module, attr, method_name, inferer,
                                pass_in_types=True):
        """
        Register an unbound method or dotted attribute path
        (allow for transience).

             E.g. np.add.reduce() can be mapped as follows:

                module=np, attr='add', method_name='reduce',
                inferrer=my_inferer
        """
        self.register_unbound_dotted(module, attr, method_name, inferer,
                                     pass_in_types)

    def register_unbound_dotted(self, module, attr, dotted_path, inferer,
                                pass_in_types=True):
        """
        Register an type function for a dotted attribute path of a value,

            E.g. my_module.my_obj.foo.bar() can be mapped as follows:

                module=my_module, attr='my_obj', dotted_path='foo.bar',
                inferrer=my_inferer
        """
        value = getattr(module, attr)
        if self.is_registered((value, dotted_path)):
            raise ValueAlreadyRegistered((value, inferer))

        self.register_value((value, dotted_path), inferer, pass_in_types)

    def get_inferer(self, value, func_type=None):
        return self.value_to_inferer[value]

    def lookup_module_attribute(self, value):
        "Return the module (or None) to which a registered value belongs"
        if self.is_registered(value) and value in self.value_to_module:
            return self.value_to_module[value]


module_registry = ModuleTypeInfererRegistry()

#----------------------------------------------------------------------------
# Dispatch Functions for the Type Inferencer
#----------------------------------------------------------------------------

def module_attribute_type(obj):
    """
    See if the object is registered to any module which might handle
    type inference on the object.
    """
    result = module_registry.lookup_module_attribute(obj)
    if result is not None:
        module, attr = result
        return typesystem.ModuleAttributeType(module=module, attr=attr)

    return None

def parse_args(call_node, arg_names):
    """
    Parse positional and keyword arguments.
    """
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

def dispatch_on_value(context, call_node, func_type):
    """
    Dispatch a call of a module attribute by value.

    For instance, a method

        def empty(shape, dtype, order):
            ...

    would be called with those arguments. Parameters not present as
    arguments in user code are None.

    Returns the result type, or None
    """
    inferer, pass_in_types = get_inferer(func_type.value)

    argnames = inspect.getargspec(inferer).args
    if argnames and argnames[0] == "context":
        argnames.pop(0)
        args = (context,)
    else:
        args = ()

    method_kwargs = parse_args(call_node, argnames)

    if pass_in_types:
        for argname, node in method_kwargs.iteritems():
            if node is not None:
                method_kwargs[argname] = get_type(node)

    return inferer(*args, **method_kwargs)

def resolve_call(context, call_node, obj_call_node, func_type):
    """
    Find the right type inferrer function for a call to an attribute
    of a certain module.


        call_node:     the original ast.Call node that we need to resolve
                       the type for

        obj_call_node: the nodes.ObjectCallNode that would replace the
                       ast.Call unless we override that with another node.

        func_type: ModuleAttributeType
            |__________> module: Python module
            |__________> attr: Attribute name
            |__________> value: Attribute value


    Returns a new AST node that should replace the ast.Call node.
    """
    result = dispatch_on_value(context, call_node, func_type)

    if result is not None and not isinstance(result, ast.AST):
        assert isinstance(result, minitypes.Type)
        type = result
        result = obj_call_node
        # result.variable = symtab.Variable(type)
        result = nodes.CoercionNode(result, type)

    return result


#----------------------------------------------------------------------------
# User-exposed functions to register type functions
#----------------------------------------------------------------------------

is_registered = module_registry.is_registered
register_inferer = module_registry.register_inferer
register_value = module_registry.register_value
get_inferer = module_registry.get_inferer
register_unbound = module_registry.register_unbound_method

def register(module, **kws):
    """
    @register(module)
    def my_type_function(arg1, ..., argN):
        ...
    """
    def decorator(inferer):
        register_inferer(module, inferer.__name__, inferer, **kws)
        return inferer

    return decorator

def register_callable(signature):
    """
    signature := FunctionType | typeset(signature *)

    @register_callable(signature)
    def my_function(...):
        ...
    """
    assert isinstance(signature, (typeset, minitypes.Type))

    def decorator(function):
        def infer(*args):
            if isinstance(signature, typeset):
                specialization = signature.from_argtypes(args)
                if specialization is None:
                    raise p

        inferer = lambda: signature.return_type
        register_value(function, inferer)
        return function
    return decorator

#----------------------------------------------------------------------------
# Registry of internal Type Functions
#----------------------------------------------------------------------------

# Register type inferrer functions
from numba.type_inference.modules import (numbamodule,
                                          numpymodule,
                                          numpyufuncs)
