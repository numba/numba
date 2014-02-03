# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ast
import string

import numba
from numba import *
from numba import nodes
from numba import pipeline
from numba import environment
from numba import numbawrapper
from numba.type_inference.module_type_inference import register_value

#------------------------------------------------------------------------
# Intrinsic Classes
#------------------------------------------------------------------------

class Intrinsic(object):

    def __init__(self, func_signature, name):
        self.func_signature = func_signature
        self.name = name

        # Register a type inference function for our intrinsic
        register_infer_intrinsic(self)

        # Build a function wrapper
        self.jitted_func = make_intrinsic(self)

    def __call__(self, *args):
        return self.jitted_func(*args)

    def emit_code(self, lfunc, builder, llvm_args):
        raise NotImplementedError

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.name == other.name and
                self.func_signature == other.func_signature)

    def __hash__(self):
        return hash((type(self), self.name, self.func_signature))


class NumbaInstruction(Intrinsic):

    def emit_code(self, lfunc, builder, llvm_args):
        return getattr(builder, self.name)(*llvm_args)


class NumbaIntrinsic(Intrinsic):

    def emit_code(self, lfunc, builder, llvm_args):
        raise NotImplementedError


def is_numba_intrinsic(value):
    return isinstance(value, Intrinsic)

numbawrapper.add_hash_by_value_type(Intrinsic)

#------------------------------------------------------------------------
# Build Intrinsic Wrapper
#------------------------------------------------------------------------

cache = {}
env = environment.NumbaEnvironment.get_environment()

def make_intrinsic(intrinsic):
    """
    Create an intrinsic function given an Intrinsic.
    """
    if intrinsic in cache:
        return cache[intrinsic]

    # NOTE: don't use numba.jit() and 'exec', it will make inspect.getsource()
    # NOTE: fail, and hence won't work in python 2.6 (since meta doesn't work
    # NOTE: there)

    # Build argument names
    nargs = len(intrinsic.func_signature.args)
    assert nargs < len(string.ascii_letters)
    argnames = ", ".join(string.ascii_letters[:nargs])

    # Build source code and environment
    args = (intrinsic.name, argnames, argnames)
    source = ("def %s(%s): return intrinsic(%s)\n" % args)
    func_globals = {'intrinsic': intrinsic}

    mod_ast = ast.parse(source)
    func_ast = mod_ast.body[0]

    # Compile
    func_env, _ = pipeline.run_pipeline2(
        env, func=None, func_ast=func_ast,
        func_signature=intrinsic.func_signature,
        function_globals=func_globals)
    jitted_func = func_env.numba_wrapper_func

    # Populate cache
    cache[intrinsic] = jitted_func
    return jitted_func

#------------------------------------------------------------------------
# Intrinsic Value Type Inference
#------------------------------------------------------------------------

class IntrinsicNode(nodes.ExprNode):
    "AST Node representing a reference to an intrinsic"

    _fields = ['args']

    def __init__(self, intrinsic, args):
        self.intrinsic = intrinsic
        self.type = self.intrinsic.func_signature.return_type
        self.args = list(args)


def register_infer_intrinsic(intrinsic):
    def infer(*args):
        return IntrinsicNode(intrinsic, args)

    register_value(intrinsic, infer,
                   pass_in_types=False,
                   can_handle_deferred_types=True)

#------------------------------------------------------------------------
# User Exposed Functionality
#------------------------------------------------------------------------

def declare_intrinsic(func_signature, name):
    """
    Declare an intrinsic, e.g.

    >>> declare_intrinsic(void(), "llvm.debugtrap")
    """
    return NumbaIntrinsic(func_signature, name)

def declare_instruction(func_signature, name):
    """
    Declare an instruction, e.g.

    >>> declare_instruction(int32(int32, int32), "add")

    The llvm.core.Builder instruction with the given name will be used.
    """
    return NumbaInstruction(func_signature, name)
