import ast
import inspect
import types

import numba
from numba import *
from numba.minivect import minitypes
from numba import typesystem, symtab, error, nodes
from numba.typesystem import get_type

import numpy.random
import numpy as np

import logging

debug = False
#debug = True

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)
if debug:
    logger.setLevel(logging.DEBUG)

class ValueAlreadyRegistered(error.NumbaError):
    """
    Raised when a type inferer is registered multiple times for the same
    value.
    """

class ModuleTypeInfererRegistry(object):
    "Builds the module type inferers for the modules we can handle"

    def __init__(self):
        super(ModuleTypeInfererRegistry, self).__init__()

        # { value : (module, attr, inferer, pass_in_types_instead_of_nodes }
        self.value_to_inferer = {}

        # { (value, unbound_dotted_path) : inferer }
        self.unbound_dotted = {}

        # { (type, bound_dotted_path) : inferer }
        self.bound_dotted = {}


    def is_registered(self, value):
        try:
            hash(value)
        except TypeError:
            return False # unhashable object
        else:
            return value in self.value_to_inferer

    def register_inferer(self, module, attr, inferer):
        value = getattr(module, attr)
        if self.is_registered(value):
            raise ValueAlreadyRegistered((value, module, inferer))

        self.value_to_inferer[value] = (module, attr, inferer, True)

    def register_unbound_method(self, value, method_name, inferer):
        self.register_unbound_dotted(value, method_name, inferer)

    def register_unbound_dotted(self, value, dotted_path, inferer):
        if self.is_registered(value):
            raise ValueAlreadyRegistered((value, inferer))

        self.unbound_dotted[value, dotted_path] = inferer

    def get_inferer(self, value):
        return self.value_to_inferer[value]

    def module_attribute_type(self, value):
        if self.is_registered(value):
            module, attr, inferer = self.get_inferer(value)
            return typesystem.ModuleAttributeType(module=module, attr=attr)

        return None

module_registry = ModuleTypeInfererRegistry()

is_registered = module_registry.is_registered
register_inferer = module_registry.register_inferer
get_inferer = module_registry.get_inferer

def register(module):
    def decorator(inferer):
        register_inferer(module, inferer.__name__, inferer)
        return inferer

    return decorator

def module_attribute_type(obj):
    """
    See if the object is registered to any module which might handle
    type inference on the object.
    """
    if is_registered(obj):
        module, attr, inferer, pass_in_types = get_inferer(obj)
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
    module, attr, inferer, pass_in_types = get_inferer(func_type.value)

    argnames = inspect.getargspec(inferer).args
    if argnames[0] == "context":
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


# Register type inferrer functions
from numba.type_inference.modules import (numbamodule,
                                          numpymodule)
