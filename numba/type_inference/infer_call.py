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
from numba import error, transforms, closure, control_flow, visitors, nodes
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

debug = False
#debug = True

def resolve_function(func_type, func_name):
    "Get a function object given a function name"
    func = None

    if func_type.is_builtin:
        func = getattr(builtins, func_name)
    elif func_type.is_global:
        func = func_type.value
    elif func_type.is_module_attribute:
        func = getattr(func_type.module, func_type.attr)
    elif func_type.is_autojit_function:
        func = func_type.autojit_func
    elif func_type.is_jit_function:
        func = func_type.jit_func

    return func
