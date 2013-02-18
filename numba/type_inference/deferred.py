import ast
import math
import cmath
import copy
import opcode
import types
import __builtin__ as builtins
from itertools import imap, izip

import numba
from numba import *
from numba import error, transforms, closures, control_flow, visitors, nodes
from numba.type_inference import module_type_inference
from numba.minivect import minierror, minitypes
from numba import translate, utils, typesystem
from numba.control_flow import ssa
from numba.typesystem.ssatypes import kosaraju_strongly_connected
from numba.symtab import Variable
from numba import stdio_util, function_util
from numba.typesystem import is_obj, promote_closest
from numba.utils import dump

import llvm.core
import numpy
import numpy as np

import logging


def create_deferred(type_inferer, node, deferred_cls):
    "Create a deferred type for an AST node"
    variable = Variable(None)
    deferred_type = deferred_cls(variable, type_inferer, node)
    variable.type = deferred_type
    node.variable = variable
    return deferred_type

def create_deferred_call(type_inferer, arg_types, call_node):
    "Set the ast.Call as uninferable for now"
    deferred_type = create_deferred(type_inferer, call_node,
                                    typesystem.DeferredCallType)
    for arg, arg_type in zip(call_node.args, arg_types):
        if arg_type.is_unresolved:
            deferred_type.dependences.append(arg)

    deferred_type.update()
    return call_node
