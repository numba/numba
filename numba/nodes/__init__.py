import ast
import ctypes

import numba
import numba.functions
from numba import function_util
from numba import *
from numba.symtab import Variable
from numba import typesystem
from numba import utils, translate, error
from numba.minivect import minitypes, minierror

import llvm.core

context = utils.get_minivect_context()

#----------------------------------------------------------------------------
# Utility Functions
#----------------------------------------------------------------------------

def objconst(obj):
    return const(obj, object_)

def const(obj, type):
    if typesystem.is_obj(type):
        node = ObjectInjectNode(obj, type)
    else:
        node = ConstNode(obj, type)

    return node

def call_pyfunc(py_func, args):
    "Generate an object call for a python function given during compilation time"
    func = ObjectInjectNode(py_func)
    return ObjectCallNode(None, func, args)

def index(node, constant_index, load=True, type=int_):
    if load:
        ctx = ast.Load()
    else:
        ctx = ast.Store()

    index = ast.Index(ConstNode(constant_index, type))
    index.type = type
    index.variable = Variable(type)
    return ast.Subscript(value=node, slice=index, ctx=ctx)


printing = False

def inject_print(context, module, node):
    node = function_util.external_call(context, module, 'PyObject_Str',
                                       args=[node])
    node = function_util.external_call(context, module, 'puts',
                                       args=[node])
    return node

def print_(translator, node):
    global printing
    if printing:
        return

    printing = True

    node = inject_print(translator.context, translator.llvm_module, node)
    node = translator.ast.pipeline.late_specializer(node)
    translator.visit(node)

    printing = False

def print_llvm(translator, type, llvm_value):
    return print_(translator, LLVMValueRefNode(type, llvm_value))

def is_name(node):
    """
    Returns whether the given node is a Name
    """
    if isinstance(node, CloneableNode):
        node = node.node
    return isinstance(node, ast.Name)

#----------------------------------------------------------------------------
# Imports
#----------------------------------------------------------------------------

# NOTE: the order of the imports is important!

from numba.nodes.basenodes import *

from numba.nodes.constnodes import *
from numba.nodes.coercionnodes import *
from numba.nodes.tempnodes import *

from numba.nodes.cfnodes import *
from numba.nodes.excnodes import *

from numba.nodes.callnodes import *
from numba.nodes.numpynodes import *
from numba.nodes.extnodes import *
from numba.nodes.closurenodes import *

from numba.nodes.usernode import *

from numba.nodes.pointernodes import *
from numba.nodes.structnodes import *
from numba.nodes.objectnodes import *
from numba.nodes.llvmnodes import *

from numba.nodes.utils import *
