# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
from numba.nodes import *

class FunctionCallNode(ExprNode):
    _attributes = ['signature', 'type', 'name']

    def __init__(self, signature, args, name=''):
        self.signature = signature
        self.type = signature.return_type
        self.name = name
        self.original_args = args

class NativeCallNode(FunctionCallNode):

    _attributes = FunctionCallNode._attributes + ['llvm_func_name']
    _fields = ['args']

    def __init__(self, signature, args, llvm_func, py_func=None,
                 badval=None, goodval=None,
                 exc_type=None, exc_msg=None, exc_args=None,
                 skip_self=False, **kw):
        super(NativeCallNode, self).__init__(signature, args, **kw)
        self.llvm_func = llvm_func
        self.llvm_func_name = getattr(llvm_func, 'name', None)
        self.py_func = py_func
        self.skip_self = skip_self
        self.type = signature.return_type
        self.coerce_args()

        self.badval = badval
        self.goodval = goodval
        self.exc_type = exc_type
        self.exc_msg = exc_msg
        self.exc_args = exc_args

    def coerce_args(self):
        self.args = list(self.original_args)
        for i, dst_type in enumerate(self.signature.args[self.skip_self:]):
            arg = self.args[i]
            self.args[i] = CoercionNode(arg, dst_type,
                                        name='func_%s_arg%d' % (self.name, i))

    def __repr__(self):
        if self.llvm_func:
            name = self.llvm_func.name
        elif self.name:
            name = self.name
        else:
            name = "<unknown(%s)>" % self.signature

        return "%s(%s)" % (name, ", ".join(str(arg) for arg in self.args))

class NativeFunctionCallNode(NativeCallNode):
    """
    Call a function which is given as a node
    """

    _fields = ['function', 'args']

    def __init__(self, signature, function_node, args, **kw):
        super(NativeFunctionCallNode, self).__init__(signature, args, None,
                                                     None, **kw)
        self.function = function_node

class LLMacroNode (NativeCallNode):
    '''
    Inject a low-level macro in the function at the call site.

    Low-level macros are Python functions that take a FunctionCache
    instance, a LLVM builder instance, and a set of arguments,
    construct LLVM code, and return some kind of LLVM value result.
    The passed signature should reflect the Numba types of the
    expected input parameters, and the type of the resulting value
    (this does not restrict polymorphism at the LLVM type level in the
    macro expansion function).
    '''

    _fields = ['macro', 'args']

    def __init__(self, signature, macro, *args, **kw):
        super(LLMacroNode, self).__init__(signature, args, None, None, **kw)
        self.macro = macro

class LLVMExternalFunctionNode(ExprNode):
    '''For calling an external llvm function where you only have the
    signature and the function name.
    '''
    def __init__(self, signature, fname):
        super(LLVMExternalFunctionNode, self).__init__(signature=signature,
                                                       fname=fname)

class LLVMIntrinsicNode(NativeCallNode):
    "Call an llvm intrinsic function"

    def __init__(self, signature, args, func_name, **kw):
        super(LLVMIntrinsicNode, self).__init__(signature, args, None, **kw)
        self.func_name = func_name

class MathCallNode(NativeCallNode):
    "Low level call a libc math function"

class PointerCallNode(NativeCallNode):
    "Call a ctypes function"

    _fields = NativeCallNode._fields + ['function']

    def __init__(self, signature, args, pointer, py_func=None, **kw):
        super(PointerCallNode, self).__init__(signature, args, None,
                                             py_func, **kw)
        self.pointer = pointer
        self.function = ConstNode(self.pointer, signature.pointer())


class ObjectCallNode(FunctionCallNode):
    _fields = ['function', 'args_tuple', 'kwargs_dict']

    def __init__(self, signature, func, args, keywords=None, py_func=None, **kw):
        if py_func and not kw.get('name', None):
            kw['name'] = py_func.__name__
        if signature is None:
            signature = numba.function(object_, [object_] * len(args))
            if keywords:
                signature.args.extend([object_] * len(keywords))

        super(ObjectCallNode, self).__init__(signature, args)
        assert func is not None
        self.function = func
        self.py_func = py_func

        self.args_tuple = ast.Tuple(elts=list(args), ctx=ast.Load())
        self.args_tuple.variable = Variable(
                typesystem.tuple_(object_, size=len(args)))

        if keywords:
            keywords = [(ConstNode(k.arg), k.value) for k in keywords]
            keys, values = zip(*keywords)
            self.kwargs_dict = ast.Dict(list(keys), list(values))
            self.kwargs_dict.variable = Variable(object_)
        else:
            self.kwargs_dict = NULL_obj

        self.type = signature.return_type

    def __repr__(self):
        return 'objcall(%s, %s)' % (self.function, self.original_args)

class ComplexConjugateNode(ExprNode):
    "mycomplex.conjugate()"

    _fields = ['complex_node']

    def __init__(self, complex_node, **kwargs):
        super(ComplexConjugateNode, self).__init__(**kwargs)
        self.complex_node = complex_node
