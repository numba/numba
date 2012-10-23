import types
import ctypes

import numba
from numba import pipeline
from numba import _numba_types as numba_types
from numba.minivect import minitypes

from numba import *

def validate_method(py_func, sig):
    if isinstance(py_func, types.FunctionType):
        nargs = py_func.func_code.co_argcount - 1
        if len(sig.args) != nargs:
            raise error.NumbaError(
                "Expected %d argument types in function "
                "%s (don't include 'self')" % (nargs, py_func.__name__))


def process_method(ext_type, method, default_signature,
                   is_static=False, is_class=False):
    if isinstance(method, minitypes.Function):
        # @double(...)
        # def func(self, ...): ...
        sig = method.signature
        validate_method(method.py_func, sig)
        argtypes = (ext_type,) + sig.args
        restype = sig.return_type
        method = method.py_func
    elif isinstance(method, types.FunctionType):
        validate_method(method, default_signature or object_())
        if default_signature:
            restype, argtypes = (default_signature.return_type,
                                 (ext_type,) + default_signature.args)
        else:
            restype, argtypes = None, (ext_type,)
    elif isinstance(method, staticmethod):
        return process_method(ext_type, method.__func__,
                              default_signature, is_static=True)
    elif isinstance(method, classmethod):
        return process_method(ext_type, method.__func__,
                              default_signature, is_class=True)
    else:
        return None, None, None

    return method, restype, argtypes


def _type_infer_method(context, ext_type, method, method_name,
                       class_dict, default_signature=None):
    method, restype, argtypes = process_method(ext_type, method,
                                               default_signature)
    if method is None:
        return

    class_dict[method_name] = method
    method.live_objects = []
    func_signature, symtab, ast = pipeline.infer_types(
                        context, method, restype, argtypes)
    ext_type.methods.append((method_name, func_signature))
    return method

def compile_extension_methods(context, py_class, ext_type, class_dict):
    method_pointers = []
    lmethods = []

    restype = None
    argtypes = (ext_type,)

    class_dict['__numba_py_class'] = py_class

    # Process __init__ first
    initfunc = class_dict.get('__init__', None)
    if initfunc is not None:
        if isinstance(initfunc, types.FunctionType):
            argtypes = [object_] * (initfunc.func_code.co_argcount - 1)
            default_signature = void(*argtypes)
        else:
            default_signature = None

        _type_infer_method(context, ext_type, initfunc, '__init__',
                           class_dict, default_signature)

    # Update native attributes before compiling methods
    attrs = dict((name, var.type) for name, var in ext_type.symtab.iteritems())
    ext_type.attribute_struct = numba.struct(**attrs)

    # Infer types for all other methods
    # We need all the types before compiling since methods can call each other
    for method_name, method in class_dict.iteritems():
        if method_name in ('__new__', '__init__') or method is None:
            continue

        _type_infer_method(context, ext_type, method, method_name, class_dict)

    # TODO: patch method call types

    # Compile methods
    for method_name, func_signature in ext_type.methods:
        method = class_dict[method_name]
        # Don't use compile_after_type_inference, re-infer, since we may
        # have inferred some return types
        # TODO: delayed types and circular calls/variable assignments
        sig, translator, wrapper = pipeline.compile(context, method,
                                                    func_signature.return_type,
                                                    func_signature.args)
        lmethods.append(translator.lfunc)
        method_pointers.append(translator.lfunc_pointer)
        class_dict[method_name] = wrapper

    return method_pointers, lmethods

def _create_descr(attr_name):
    def _get(self):
        return getattr(self._numba_attrs, attr_name)
    def _set(self, value):
        return setattr(self._numba_attrs, attr_name, value)
    return property(_get, _set)

def inject_descriptors(context, py_class, ext_type, class_dict):
    for attr_name, attr_type in ext_type.symtab.iteritems():
        descriptor = _create_descr(attr_name)
        class_dict[attr_name] = descriptor

def vtab_name(field_name):
    if field_name.startswith("__") and field_name.endswith("__"):
        field_name = '__numba_' + field_name.strip("_")
    return field_name

def build_vtab(vtab_type, method_pointers):
    assert len(method_pointers) == len(vtab_type.fields)

    vtab_type = numba.struct([(vtab_name(field_name), field_type)
                                  for field_name, field_type in vtab_type.fields])
    vtab_ctype = vtab_type.to_ctypes()

    methods = []
    for method_pointer, (field_name, field_type) in zip(method_pointers,
                                                        vtab_type.fields):
        cmethod = field_type.to_ctypes()(method_pointer)
        methods.append(cmethod)

    vtab = vtab_ctype(*methods)
    return vtab

def create_extension(context, py_class, translator_kwargs):
    """
    Compile an extension class

        1) Create an extension Numba/minivect type holding a symtab
        2) Capture native extension attributes from __init__ in the symtab
        3) Type infer all methods
        4) Compile all methods
        5) Create descriptors that wrap the native attributes
        6) Create an extension type:

          {
            PyObject_HEAD
            ...
            virtual function table (func **)
            native attributes
          }

        The virtual function table (vtab) is a ctypes structure set as
        attribute of the extension types. Objects have a direct pointer
        for efficiency.
    """
    type = numba_types.ExtensionType(py_class)
    class_dict = dict(vars(py_class))
    method_pointers, lmethods = compile_extension_methods(
                                        context, py_class, type, class_dict)
    type.vtab_type = struct(type.methods)
    inject_descriptors(context, py_class, type, class_dict)

    vtab = build_vtab(type.vtab_type, method_pointers)

    extension_type = extension_types.create_new_extension_type(
            py_class.__name__, py_class.__bases__, class_dict,
            type.attribute_struct, vtab,
            lmethods, method_pointers)
    return extension_type