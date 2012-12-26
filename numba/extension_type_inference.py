import types
import ctypes
import logging

import numba
from numba import pipeline, error, symtab
from numba import typesystem
from numba.minivect import minitypes

from numba import *

logger = logging.getLogger(__name__)

def validate_method(py_func, sig, is_static):
    assert isinstance(py_func, types.FunctionType)
    nargs = py_func.func_code.co_argcount - 1 + is_static
    if len(sig.args) != nargs:
        raise error.NumbaError(
            "Expected %d argument types in function "
            "%s (don't include 'self')" % (nargs, py_func.__name__))

def get_signature(ext_type, is_class, is_static, sig):
    """
    Create a signature given the user-specified signature. E.g.

        class Foo(object):
            @void()                 # becomes: void(ext_type(Foo))
            def method(self): ...
    """
    if is_static:
        leading_arg_types = ()
    elif is_class:
        leading_arg_types = (object_,)
    else:
        leading_arg_types = (ext_type,)

    argtypes = leading_arg_types + sig.args
    restype = sig.return_type
    return minitypes.FunctionType(return_type=restype, args=argtypes)

def _process_signature(ext_type, method, default_signature,
                       is_static=False, is_class=False):
    """
    Verify a method signature.

    Returns a Method object and the resolved signature.
    """
    if isinstance(method, minitypes.Function):
        # @double(...)
        # def func(self, ...): ...
        return _process_signature(ext_type, method.py_func,
                                  method.signature, is_static, is_class)
    elif isinstance(method, types.FunctionType):
        if default_signature is None:
            # TODO: construct dependency graph, toposort, type infer
            # TODO: delayed types
            raise error.NumbaError(
                "Method '%s' does not have signature" % (method.__name__,))
        validate_method(method, default_signature or object_(), is_static)
        if default_signature is None:
            default_signature = minitypes.FunctionType(return_type=None,
                                                       args=[])
        sig = get_signature(ext_type, is_class, is_static, default_signature)
        return Method(method, is_class, is_static), sig.return_type, sig.args
    else:
        if isinstance(method, staticmethod):
            is_static = True
        elif isinstance(method, classmethod):
            is_class = True
        else:
            return None, None, None

        method, restype, argtypes = _process_signature(
                        ext_type, method.__func__, default_signature,
                        is_static=is_static, is_class=is_class)

    return method, restype, argtypes

class Method(object):
    """
    py_func: the python 'def' function
    """

    def __init__(self, py_func, is_class, is_static):
        self.py_func = py_func
        py_func.live_objects = []
        self.is_class = is_class
        self.is_static = is_static
        self.name = py_func.__name__

    def result(self, py_func):
        if self.is_class:
            return classmethod(py_func)
        elif self.is_static:
            return staticmethod(py_func)
        else:
            return py_func

def _process_method_signatures(class_dict, ext_type):
    "Process all method signatures in order"
    for method_name, method in class_dict.iteritems():
        default_signature = None
        if (method_name == '__init__' and
                isinstance(method, types.FunctionType)):
            argtypes = [object_] * (method.func_code.co_argcount - 1)
            default_signature = void(*argtypes)

        method, restype, argtypes = _process_signature(ext_type, method,
                                                       default_signature)
        if method is None:
            continue

        signature = typesystem.ExtMethodType(
                    return_type=restype, args=argtypes, name=method.name,
                    is_class=method.is_class, is_static=method.is_static)
        ext_type.add_method(method_name, signature)
        class_dict[method_name] = method

def _type_infer_method(context, ext_type, method, method_name, class_dict,
                       llvm_module):
    if method_name not in ext_type.methoddict:
        return

    signature = ext_type.get_signature(method_name)
    restype, argtypes = signature.return_type, signature.args

    class_dict[method_name] = method
    func_signature, symtab, ast = pipeline.infer_types(
                        context, method.py_func, restype, argtypes)
    ext_type.add_method(method_name, func_signature)

def _type_infer_init_method(context, class_dict, ext_type, llvm_module):
    initfunc = class_dict.get('__init__', None)
    if initfunc is None:
        return

    _type_infer_method(context, ext_type, initfunc, '__init__', class_dict,
                       llvm_module)

def _type_infer_methods(context, class_dict, ext_type, llvm_module):
    for method_name, method in class_dict.iteritems():
        if method_name in ('__new__', '__init__') or method is None:
            continue

        _type_infer_method(context, ext_type, method, method_name, class_dict,
                           llvm_module)

def _compile_methods(class_dict, context, ext_type, lmethods, method_pointers,
                     llvm_module):
    parent_method_pointers = getattr(
                    ext_type.py_class, '__numba_method_pointers', None)
    for i, (method_name, func_signature) in enumerate(ext_type.methods):
        if method_name not in class_dict:
            # Inherited method
            assert parent_method_pointers is not None
            name, p = parent_method_pointers[i]
            assert name == method_name
            method_pointers.append((method_name, p))
            continue

        method = class_dict[method_name]
        # Don't use compile_after_type_inference, re-infer, since we may
        # have inferred some return types
        # TODO: delayed types and circular calls/variable assignments
        sig, translator, wrapper = pipeline.compile(context, method.py_func,
                                                    func_signature.return_type,
                                                    func_signature.args)
        lmethods.append(translator.lfunc)
        method_pointers.append((method_name, translator.lfunc_pointer))
        class_dict[method_name] = method.result(wrapper)

def _construct_native_attribute_struct(ext_type):
    attrs = dict((name, var.type) for name, var in ext_type.symtab.iteritems())
    if ext_type.attribute_struct is None:
        # No fields to inherit
        ext_type.attribute_struct = numba.struct(**attrs)
    else:
        # Inherit fields from parent
        fields = []
        for name, variable in ext_type.symtab.iteritems():
            if name not in ext_type.attribute_struct.fielddict:
                fields.append((name, variable.type))
                ext_type.attribute_struct.fielddict[name] = variable.type

        # Sort fields by rank
        fields = numba.struct(fields).fields
        ext_type.attribute_struct.fields.extend(fields)

def compile_extension_methods(context, py_class, ext_type, class_dict, llvm_module):
    """
    Compile extension methods:

        1) Process signatures such as @void(double)
        2) Infer native attributes through type inference on __init__
        3) Path the extension type with a native attributes struct
        4) Infer types for all other methods
        5) Update the ext_type with a vtab type
        6) Compile all methods
    """
    method_pointers = []
    lmethods = []

    class_dict['__numba_py_class'] = py_class

    _process_method_signatures(class_dict, ext_type)
    _type_infer_init_method(context, class_dict, ext_type, llvm_module)
    _construct_native_attribute_struct(ext_type)
    _type_infer_methods(context, class_dict, ext_type, llvm_module)

    # TODO: patch method call types

    # Set vtab type before compiling
    ext_type.vtab_type = struct([(field_name, field_type.pointer())
                                    for field_name, field_type in ext_type.methods])
    _compile_methods(class_dict, context, ext_type, lmethods, method_pointers,
                     llvm_module)
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

def is_numba_class(cls):
    return hasattr(cls, '__numba_struct_type')

def verify_base_class_compatibility(cls, struct_type, vtab_type):
    bases = [cls]
    for base in cls.__bases__:
        if is_numba_class(base):
            attr_prefix = base.__numba_struct_type.is_prefix(struct_type)
            method_prefix = base.__numba_vtab_type.is_prefix(vtab_type)
            if not attr_prefix or not method_prefix:
                raise error.NumbaError(
                            "Multiple incompatible base classes found: "
                            "%s and %s" % (base, bases[-1]))

            bases.append(base)

def inherit_attributes(ext_type, class_dict):
    "Inherit attributes and methods from superclasses"
    cls = ext_type.py_class
    if not is_numba_class(cls):
        # superclass is not a numba class
        return

    struct_type = cls.__numba_struct_type
    vtab_type = cls.__numba_vtab_type
    verify_base_class_compatibility(cls, struct_type, vtab_type)

    # Inherit attributes
    ext_type.attribute_struct = numba.struct(struct_type.fields)
    for field_name, field_type in ext_type.attribute_struct.fields:
        ext_type.symtab[field_name] = symtab.Variable(field_type,
                                                      promotable_type=False)

    # Inherit methods
    for method_name, method_type in vtab_type.fields:
        func_signature = method_type.base_type
        args = list(func_signature.args)
        if not (func_signature.is_class or func_signature.is_static):
            args[0] = ext_type
        func_signature = func_signature.return_type(*args)
        ext_type.add_method(method_name, func_signature)

    ext_type.parent_attr_struct = struct_type
    ext_type.parent_vtab_type = vtab_type

def vtab_name(field_name):
    if field_name.startswith("__") and field_name.endswith("__"):
        field_name = '__numba_' + field_name.strip("_")
    return field_name

def build_vtab(vtab_type, method_pointers):
    assert len(method_pointers) == len(vtab_type.fields)

    vtab_ctype = numba.struct(
        [(vtab_name(field_name), field_type)
            for field_name, field_type in vtab_type.fields]).to_ctypes()

    methods = []
    for (method_name, method_pointer), (field_name, field_type) in zip(
                                        method_pointers, vtab_type.fields):
        assert method_name == field_name
        method_type_p = field_type.to_ctypes()
        cmethod = ctypes.cast(ctypes.c_void_p(method_pointer), method_type_p)
        methods.append(cmethod)

    vtab = vtab_ctype(*methods)
    return vtab, vtab_type

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
    
    llvm_module = translator_kwargs.pop('llvm_module')
    assert llvm_module is not None
    ext_type = typesystem.ExtensionType(py_class)
    class_dict = dict(vars(py_class))
    inherit_attributes(ext_type, class_dict)

    method_pointers, lmethods = compile_extension_methods(
                                        context, py_class, ext_type, class_dict,
                                        llvm_module)
    inject_descriptors(context, py_class, ext_type, class_dict)

    vtab, vtab_type = build_vtab(ext_type.vtab_type, method_pointers)

    logger.info("struct: %s" % ext_type.attribute_struct)
    logger.info("ctypes struct: %s" % ext_type.attribute_struct.to_ctypes())
    extension_type = extension_types.create_new_extension_type(
            py_class.__name__, py_class.__bases__, class_dict,
            ext_type, vtab, vtab_type,
            lmethods, method_pointers)
    return extension_type