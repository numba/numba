"""
Module providing some test utilities.
"""

import os
import sys

import miniast
import specializers
import minitypes
import miniutils
import codegen
import treepath

from minitypes import *
from miniutils import *
from specializers import specializers as sps
from ctypes_conversion import get_data_pointer, convert_to_ctypes

def getcontext():
    return miniast.CContext()

def get_llvm_context():
    context = miniast.LLVMContext()
    context.shape_type = minitypes.npy_intp.pointer()
    context.strides_type = context.shape_type
    return context

def build_vars(*types):
    return [b.variable(type, 'op%d' % i) for i, type in enumerate(types)]

def build_function(variables, body, name=None):
    qualify = lambda type: type.qualify("const", "restrict")
    func = context.astbuilder.build_function(variables, body, name)
    func.shape.type = qualify(func.shape.type)
    for arg in func.arguments:
        if arg.type.is_array:
            arg.data_pointer.type = qualify(arg.data_pointer.type)
            arg.strides_pointer.type = qualify(arg.strides_pointer.type)

    return func

def specialize(specializer_cls, ast, context=None, print_tree=False):
    context = context or getcontext()
    return miniutils.specialize(context, specializer_cls, ast,
                                print_tree=print_tree)

def run(specializers, ast):
    context = getcontext()
    for result in context.run(ast, specializers):
        _, specialized_ast, _, (proto, impl) = result
        yield specialized_ast, impl

def toxml(function):
    return xmldumper.XMLDumper(context).visit(function)

# Convenience variables
context = getcontext()
b = context.astbuilder