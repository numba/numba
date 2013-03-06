from llvm import core as _lc
import numba
from numba import pipeline, error, symtab, typesystem
from numba import typesystem
from numba.exttypes import extension_types
from numba.exttypes.extension_type_inference import inherit_attributes, process_class_attribute_types, compile_extension_methods, inject_descriptors, build_vtab, logger
from numba.minivect import minitypes

from numba.exttypes import logger

#------------------------------------------------------------------------
# Build Extension Type (@jit)
#------------------------------------------------------------------------

def jit_extension_class(py_class, translator_kwargs, env):
    llvm_module = translator_kwargs.get('llvm_module', None)
    if llvm_module is None:
        llvm_module = _lc.Module.new('tmp.extension_class.%X' % id(py_class))
        translator_kwargs['llvm_module'] = llvm_module

    return create_extension(env, py_class, translator_kwargs)

def create_extension(env, py_class, flags):
    """
    Compile an extension class given the NumbaEnvironment and the Python
    class that contains the functions that are to be compiled.
    """
    flags.pop('llvm_module', None)

    ext_type = typesystem.ExtensionType(py_class)
    class_dict = dict(vars(py_class))

    inherit_attributes(ext_type, class_dict)
    process_class_attribute_types(ext_type, class_dict)

    method_pointers, lmethods = compile_extension_methods(
            env, py_class, ext_type, class_dict, flags)
    inject_descriptors(env, py_class, ext_type, class_dict)

    vtab, vtab_type = build_vtab(ext_type.vtab_type, method_pointers)

    logger.debug("struct: %s" % ext_type.attribute_struct)
    logger.debug("ctypes struct: %s" % ext_type.attribute_struct.to_ctypes())

    extension_type = extension_types.create_new_extension_type(
            py_class.__name__, py_class.__bases__, class_dict,
            ext_type, vtab, vtab_type,
            lmethods, method_pointers)
    return extension_type

#------------------------------------------------------------------------
# Build Dynamic Extension Type (@autojit)
#------------------------------------------------------------------------

def create_dynamic_extension(env, py_class, flags):
    """
    Compile an extension class given the NumbaEnvironment and the Python
    class that contains the functions that are to be compiled.
    """
    raise NotImplementedError
