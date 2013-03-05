"""
Module that creates wrapper around llvm functions. The wrapper is callable
from Python.
"""

import ast
import ctypes

import llvm.core

from numba import *
from numba import nodes
from numba import closures
from numba import typesystem
from numba import extension_types
from numba.codegen import translate

from numba.functions import keep_alive
from numba.symtab import Variable
from numba.typesystem import is_obj, promote_closest, get_type

#------------------------------------------------------------------------
# Create a NumbaFunction (numbafunction.c)
#------------------------------------------------------------------------

def _create_methoddef(py_func, func_name, func_doc, func_pointer):
    """
    Create a PyMethodDef ctypes struct.

    struct PyMethodDef {
         const char  *ml_name;   /* The name of the built-in function/method */
         PyCFunction  ml_meth;   /* The C function that implements it */
         int      ml_flags;      /* Combination of METH_xxx flags, which mostly
                                    describe the args expected by the C func */
         const char  *ml_doc;    /* The __doc__ attribute, or NULL */
    };
    """
    PyMethodDef = struct([('name', c_string_type),
                          ('method', void.pointer()),
                          ('flags', int_),
                          ('doc', c_string_type)])
    c_PyMethodDef = PyMethodDef.to_ctypes()

    PyCFunction_NewEx = ctypes.pythonapi.PyCFunction_NewEx
    PyCFunction_NewEx.argtypes = [ctypes.POINTER(c_PyMethodDef),
                                  ctypes.py_object,
                                  ctypes.c_void_p]
    PyCFunction_NewEx.restype = ctypes.py_object

    # It is paramount to put these into variables first, since every
    # access may return a new string object!
    keep_alive(py_func, func_name)
    keep_alive(py_func, func_doc)

    methoddef = c_PyMethodDef()
    if PY3:
        if func_name is not None:
            func_name = func_name.encode('utf-8')
        if func_doc is not None:
            func_doc = func_doc.encode('utf-8')

    methoddef.name = func_name
    methoddef.doc = func_doc
    methoddef.method = ctypes.c_void_p(func_pointer)
    methoddef.flags = 1 # METH_VARARGS

    return methoddef

def numbafunction_new(py_func, func_name, func_doc, module_name, func_pointer,
                      wrapped_lfunc_pointer, wrapped_signature):
    "Create a NumbaFunction (numbafunction.c)"
    methoddef = _create_methoddef(py_func, func_name, func_doc, func_pointer)

    keep_alive(py_func, methoddef)
    keep_alive(py_func, module_name)

    wrapper = extension_types.create_function(methoddef, py_func,
                                              wrapped_lfunc_pointer,
                                              wrapped_signature, module_name)
    return methoddef, wrapper


#------------------------------------------------------------------------
# Ctypes wrapping
#------------------------------------------------------------------------

def get_ctypes_func(self, llvm=True):
    import ctypes
    sig = self.func_signature
    restype = typesystem.convert_to_ctypes(sig.return_type)

    # FIXME: Switch to PYFUNCTYPE so it does not release the GIL.
    #
    #    prototype = ctypes.CFUNCTYPE(restype,
    #                                 *[_types.convert_to_ctypes(x)
    #                                       for x in sig.args])
    prototype = ctypes.PYFUNCTYPE(restype,
                                 *[typesystem.convert_to_ctypes(x)
                                       for x in sig.args])


    if hasattr(restype, 'make_ctypes_prototype_wrapper'):
        # See numba.utils.ComplexMixin for an example of
        # make_ctypes_prototype_wrapper().
        prototype = restype.make_ctypes_prototype_wrapper(prototype)

    if llvm:
        # July 10, 2012: PY_CALL_TO_LLVM_CALL_MAP is removed recent commit.
        #
        #    PY_CALL_TO_LLVM_CALL_MAP[self.func] = \
        #        self.build_call_to_translated_function
        return prototype(self.lfunc_pointer)
    else:
        return prototype(self.func)

#------------------------------------------------------------------------
# NumbaFunction Wrapping
#------------------------------------------------------------------------

def fake_pyfunc(self, args):
    "PyObject *(*)(PyObject *self, PyObject *args)"
    pass

def get_closure_scope(func_signature, func_obj):
    """
    Retrieve the closure from the NumbaFunction from the func_closure
    attribute.

        func_signature:
            signature of closure function

        func_obj:
            LLVM Value referencing the closure function as a Python object
    """
    closure_scope_type = func_signature.args[0]
    offset = extension_types.numbafunc_closure_field_offset
    closure = nodes.LLVMValueRefNode(void.pointer(), func_obj)
    closure = nodes.CoercionNode(closure, char.pointer())
    closure_field = nodes.pointer_add(closure, nodes.const(offset, size_t))
    closure_field = nodes.CoercionNode(closure_field,
                                       closure_scope_type.pointer())
    closure_scope = nodes.DereferenceNode(closure_field)
    return closure_scope

def build_wrapper_function_ast(env, llvm_module):
    """
    Build AST for LLVM function wrapper.

        lfunc: LLVM function to wrap
        llvm_module: module the wrapper is being defined in
    """
    func = env.crnt.func
    func_signature = env.crnt.func_signature
    func_name = env.crnt.func_name

    # Insert external declaration
    lfunc = llvm_module.get_or_insert_function(
        func_signature.to_llvm(env.context),
        env.crnt.lfunc.name)

    # Build AST
    wrapper = nodes.FunctionWrapperNode(lfunc,
                                        func_signature,
                                        func,
                                        fake_pyfunc,
                                        func_name)

    error_return = ast.Return(nodes.CoercionNode(nodes.NULL_obj,
                                                 object_))

    is_closure = bool(closures.is_closure_signature(func_signature))
    nargs = len(func_signature.args) - is_closure

    # Call wrapped function with unpacked object arguments (delay actual arguments)
    args = [nodes.LLVMValueRefNode(object_, None)
                for i in range(nargs)]

    if is_closure:
        # Insert m_self as scope argument type
        closure_scope = get_closure_scope(func_signature, lfunc.args[0])
        args.insert(0, closure_scope)

    func_call = nodes.NativeCallNode(func_signature, args, lfunc)

    if not is_obj(func_signature.return_type):
        # Check for error using PyErr_Occurred()
        func_call = nodes.PyErr_OccurredNode(func_call)

    # Coerce and return result
    if func_signature.return_type.is_void:
        wrapper.body = func_call
        result_node = nodes.ObjectInjectNode(None)
    else:
        wrapper.body = None
        result_node = func_call

    wrapper.return_result = ast.Return(value=nodes.CoercionNode(result_node,
                                                                object_))


    # Update wrapper
    wrapper.error_return = error_return
    wrapper.cellvars = []
    wrapper.wrapped_args = args

    return wrapper

def build_wrapper_translation(env, llvm_module=None):
    if llvm_module:
        wrapper_module = llvm_module
    else:
        wrapper_module = env.llvm_context.module

    # Create wrapper code generator and wrapper AST
    func_name = '__numba_wrapper_%s' % env.crnt.func_name
    signature = object_(void.pointer(), object_)
    symtab = dict(self=Variable(object_, is_local=True),
                  args=Variable(object_, is_local=True))

    wrapper_call = build_wrapper_function_ast(env, llvm_module=wrapper_module)

    func_env = env.crnt.inherit(
            name=func_name,
            mangled_name=None, # Force FunctionEnvironment.init()
                               # to generate a new mangled name.
            func_signature=signature,
            llvm_module=wrapper_module)
    env.translation.push_env(func_env)

    # TODO: Redo this
    try:
        from numba import pipeline
        pipeline.create_lfunc(wrapper_call, env)

        t = translate.LLVMCodeGenerator(
            env.context, fake_pyfunc, wrapper_call, signature,
            symtab, llvm_module=wrapper_module,
            locals={}, refcount_args=False,
            func_name=func_name, env=env)
        t.translate()
    finally:
        env.translation.pop()

    keep_alive(fake_pyfunc, func_env.lfunc)
    return t

def build_wrapper_function(env):
    '''
    Build a wrapper function for the currently translated function.

    Return the interpreter-level wrapper function, the LLVM wrapper function,
    and the method definition record.
    '''
    t = build_wrapper_translation(env)

    # Return a PyCFunctionObject holding the wrapper
    func_pointer = t.lfunc_pointer
    methoddef, wrapper = numbafunction_new(
            env.crnt.func,
            env.crnt.func_name,
            env.crnt.func_doc,
            env.crnt.translator.module_name,
            func_pointer,                       # Wrapper
            env.crnt.translator.lfunc_pointer,  # Wrapped
            env.crnt.func_signature)

    return wrapper, t.lfunc, methoddef

def build_wrapper_module(env):
    '''
    Build a wrapper function for the currently translated
    function, and return a tuple containing the separate LLVM
    module, and the LLVM wrapper function.
    '''
    llvm_module = lc.Module.new('%s_wrapper_module' % env.crnt.mangled_name)
    t = build_wrapper_translation(env, llvm_module=llvm_module)
    logger.debug('Wrapper module: %s' % llvm_module)
    return llvm_module, t.lfunc
