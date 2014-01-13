# -*- coding: utf-8 -*-
"""
Support for type functions for external code.

See modules/numpy*.py for type inference for NumPy.
"""
from __future__ import print_function, division, absolute_import

import ast
import inspect

import numba
from numba import *
from numba import typesystem, error, nodes
from numba.typesystem import get_type, typeset, Type

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

        # value := (typefunc, pass_in_types, pass_in_callnode)
        # function calls: (np.add)
        #     { value                               :  value }
        # unbound methods: (np.add.reduce)
        #     { (value, unbound_dotted_path, False) : value }
        # bound methods: (obj.method where type(obj) is registered)
        #     { (type, bound_dotted_path, True)     : value }
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

    def register_inferer(self, module, attr, inferer, **kwds):
        """
        Register an type function (a type inferer) for a known function value.

            E.g. np.add() can be mapped as follows:

                module=np, attr='add', inferrer=my_inferer
        """
        value = getattr(module, attr)
        if self.is_registered(value):
            raise ValueAlreadyRegistered((value, module, inferer))

        self.value_to_module[value] = (module, attr)
        self.register_value(value, inferer, **kwds)

    def register_value(self, value, inferer, pass_in_types=True,
                       pass_in_callnode=False, can_handle_deferred_types=False):
        flags = dict(
            pass_in_types=pass_in_types,
            pass_in_callnode=pass_in_callnode,
            can_handle_deferred_types=can_handle_deferred_types,
        )
        self.value_to_inferer[value] = (inferer, flags)

    def register_unbound_method(self, module, attr, method_name,
                                inferer, **kwds):
        """
        Register an unbound method or dotted attribute path
        (allow for transience).

             E.g. np.add.reduce() can be mapped as follows:

                module=np, attr='add', method_name='reduce',
                inferrer=my_inferer
        """
        self.register_unbound_dotted(module, attr, method_name, inferer,
                                     **kwds)

    def register_unbound_dotted(self, module, attr, dotted_path, inferer,
                                **kwds):
        """
        Register an type function for a dotted attribute path of a value,

            E.g. my_module.my_obj.foo.bar() can be mapped as follows:

                module=my_module, attr='my_obj', dotted_path='foo.bar',
                inferrer=my_inferer
        """
        value = getattr(module, attr)
        self.register_unbound_dotted_value(value, dotted_path, inferer, **kwds)

    def register_unbound_dotted_value(self, value, dotted_path,
                                      inferer, **kwds):
        if self.is_registered((value, dotted_path)):
            raise ValueAlreadyRegistered((value, inferer))

        self.register_value((value, dotted_path), inferer, **kwds)

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
        return typesystem.module_attribute(module=module, attr=attr)

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

def _build_arg(pass_in_types, node):
    if pass_in_types:
        return get_type(node)
    return node

def dispatch_on_value(context, call_node, func_type): # TODO: Pass in typesystem here
    """
    Dispatch a call of a module attribute by value.

    For instance, a method

        def empty(shape, dtype, order):
            ...

    would be called with those arguments. Parameters not present as
    arguments in user code are None.

    Returns the result type, or None
    """
    inferer, flags = get_inferer(func_type.value)

    # Detect needed arguments
    argspec = inspect.getargspec(inferer)

    # Pass in additional arguments (context and call_node)
    argnames = argspec.args
    if argnames and argnames[0] in ("context", "typesystem"):
        argnames.pop(0)
        # TODO: Remove this and reference in mathmodule.infer_unary_math_call
        context.env.crnt.typesystem.env = context.env
        args = [context.env.crnt.typesystem]
    else:
        args = []

    if flags['pass_in_callnode']:
        argnames.pop(0)
        args.append(call_node)

    # Parse argument names from introspection
    method_kwargs = parse_args(call_node, argnames)

    # Build keyword arguments
    for argname, node in method_kwargs.iteritems():
        if node is not None:
            method_kwargs[argname] = _build_arg(flags['pass_in_types'], node)

    if argspec.varargs and len(argnames) < len(call_node.args):
        # In the case of *args, build positional list and pass any additional
        # arguments as keywords
        extra_args = call_node.args[len(argnames):]

        args.extend(method_kwargs.pop(argname) for argname in argnames)
        args.extend(_build_arg(flags['pass_in_types'], arg) for arg in extra_args)

    if argspec.keywords:
        # Handle **kwargs
        for keyword in call_node.keywords:
            if keyword.arg not in argnames:
                method_kwargs[keyword.arg] = keyword.value

    return inferer(*args, **method_kwargs)

def resolve_call(context, call_node, obj_call_node, func_type):
    """
    Find the right type inferrer function for a call to an attribute
    of a certain module.


        call_node:     the original ast.Call node that we need to resolve
                       the type for

        obj_call_node: the nodes.ObjectCallNode that would replace the
                       ast.Call unless we override that with another node.

        func_type: module_attribute
            |__________> module: Python module
            |__________> attr: Attribute name
            |__________> value: Attribute value


    Returns a new AST node that should replace the ast.Call node.
    """
    result = dispatch_on_value(context, call_node, func_type)

    if result is not None and not isinstance(result, ast.AST):
        assert isinstance(result, Type), (Type, result)
        type = result
        result = obj_call_node
        # result.variable = symtab.Variable(type)
        result = nodes.CoercionNode(result, type)

    return result

def resolve_call_or_none(context, call_node, func_type):
    if (func_type.is_known_value and
            is_registered(func_type.value)):
        # Try the module type inferers
        new_node = nodes.call_obj(call_node, None)
        return resolve_call(context, call_node, new_node, func_type)

def can_handle_deferred(py_func):
    "Return whether the type function can handle deferred argument types"
    inferer, flags = get_inferer(py_func)
    return flags['can_handle_deferred_types']

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
    signature := function | typeset(signature *)

    @register_callable(signature)
    def my_function(...):
        ...
    """
    assert isinstance(signature, (typeset.typeset, Type))

    # convert void return type to object_ (None)
    def convert_void_to_object(sig):
        if sig.return_type == void:
            sig = sig.add('return_type', object_)
        return sig

    if isinstance(signature, typeset.typeset):
        signature = typeset.typeset([convert_void_to_object(x)
                                     for x in signature.types],
                                    name=signature.name)
    else:
        assert isinstance(signature, Type)
        signature = convert_void_to_object(signature)


    def decorator(function):
        def infer(typesystem, *args):
            if signature.is_typeset:
                specialization = signature.find_match(typesystem.promote, args)
            else:
                specialization = typeset.match(typesystem.promote, signature, args)

            if specialization is None:
                raise UnmatchedTypeError(
                        "Unmatched argument types for function '%s': %s" %
                                                    (function.__name__, args))

            assert specialization.is_function
            return specialization.return_type

        register_value(function, infer)
        return function

    return decorator


#----------------------------------------------------------------------------
# Registry of internal Type Functions
#----------------------------------------------------------------------------

# Register type inferrer functions
from numba.type_inference.modules import (numbamodule,
                                          numpymodule,
                                          numpyufuncs,
                                          builtinmodule,
                                          mathmodule)
