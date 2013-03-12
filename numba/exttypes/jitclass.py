"""
Compiling @jit extension classes works as follows:

    * Create an extension Numba/minivect type holding a symtab
    * Capture attribute types in the symtab ...

        * ... from the class attributes:

            @jit
            class Foo(object):
                attr = double

        * ... from __init__

            @jit
            class Foo(object):
                def __init__(self, attr):
                    self.attr = double(attr)

    * Type infer all methods
    * Compile all extension methods

        * Process signatures such as @void(double)
        * Infer native attributes through type inference on __init__
        * Path the extension type with a native attributes struct
        * Infer types for all other methods
        * Update the ext_type with a vtab type
        * Compile all methods

    * Create descriptors that wrap the native attributes
    * Create an extension type:

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

import numba
from numba import error
from numba import typesystem
from numba import pipeline
from numba import symtab
from numba.exttypes.utils import is_numba_class
from numba.minivect import minitypes

from numba.exttypes import logger
from numba.exttypes import virtual
from numba.exttypes import signatures
from numba.exttypes import utils
from numba.exttypes import compileclass
from numba.exttypes import extension_types

#------------------------------------------------------------------------
# Populate Extension Type with Methods
#------------------------------------------------------------------------

class JitExtensionCompiler(compileclass.ExtensionCompiler):
    """
    Compile @jit extension classes.
    """

    method_validators = signatures.jit_validators

    def inherit_method(self, method_name, slot_idx):
        """
        Inherit a method from a superclass in the vtable.

        :return: a pointer to the function.
        """
        parent_method_pointers = utils.get_method_pointers(self.py_class)

        assert parent_method_pointers is not None
        name, pointer = parent_method_pointers[slot_idx]
        assert name == method_name

        return pointer

    def compile_methods(self):
        method_pointers = []
        lmethods = []

        for i, (method_name, func_signature) in enumerate(self.ext_type.methods):
            if method_name not in self.class_dict:
                pointer = self.inherit_method(method_name, i)
                method_pointers.append((method_name, pointer))
                continue

            method = self.class_dict[method_name]
            # Don't use compile_after_type_inference, re-infer, since we may
            # have inferred some return types
            # TODO: delayed types and circular calls/variable assignments
            logger.debug(method.py_func)

            func_env = self.func_envs[method]
            pipeline.run_env(self.env, func_env, pipeline_name='compile')

            lmethods.append(func_env.lfunc)
            method_pointers.append((method_name,
                                    func_env.translator.lfunc_pointer))

            self.class_dict[method_name] = method.result(
                func_env.numba_wrapper_func)

        return method_pointers, lmethods

#------------------------------------------------------------------------
# Attribute Inheritance
#------------------------------------------------------------------------

class JitAttributesInheriter(compileclass.AttributesInheriter):
    """
    Inherit attributes and methods from parent classes.
    """

    def inherit_attributes(self, ext_type, parent_struct_type):
        ext_type.parent_attr_struct = parent_struct_type
        ext_type.attribute_table = numba.struct(parent_struct_type.fields)

        for field_name, field_type in ext_type.attribute_table.fields:
            ext_type.symtab[field_name] = symtab.Variable(field_type,
                                                          promotable_type=False)

    def inherit_methods(self, ext_type, parent_vtab_type):
        ext_type.parent_vtab_type = parent_vtab_type

        for method_name, method_type in parent_vtab_type.fields:
            func_signature = method_type.base_type
            args = list(func_signature.args)
            if not (func_signature.is_class or func_signature.is_static):
                args[0] = ext_type
            func_signature = func_signature.return_type(*args)
            ext_type.add_method(method_name, func_signature)

    def verify_base_class_compatibility(self, py_class, struct_type, vtab_type):
        "Verify that we can build a compatible class layout"
        bases = [py_class]
        for base in py_class.__bases__:
            if is_numba_class(base):
                attr_prefix = utils.get_attributes_type(base).is_prefix(struct_type)
                method_prefix = utils.get_vtab_type(base).is_prefix(vtab_type)
                if not attr_prefix or not method_prefix:
                    raise error.NumbaError(
                                "Multiple incompatible base classes found: "
                                "%s and %s" % (base, bases[-1]))

                bases.append(base)

#------------------------------------------------------------------------
# Build Attributes Struct
#------------------------------------------------------------------------

class JitAttributeBuilder(compileclass.AttributeBuilder):

    def create_descr(self, attr_name):
        """
        Create a descriptor that accesses the attribute on the ctypes struct.
        """
        def _get(self):
            return getattr(self._numba_attrs, attr_name)
        def _set(self, value):
            return setattr(self._numba_attrs, attr_name, value)
        return property(_get, _set)

#------------------------------------------------------------------------
# Build Extension Type
#------------------------------------------------------------------------

def create_extension(env, py_class, flags):
    """
    Compile an extension class given the NumbaEnvironment and the Python
    class that contains the functions that are to be compiled.
    """
    flags.pop('llvm_module', None)

    ext_type = typesystem.JitExtensionType(py_class)

    extension_compiler = JitExtensionCompiler(
        env, py_class, ext_type, flags,
        signatures.JitMethodMaker(ext_type),
        JitAttributesInheriter(),
        JitAttributeBuilder(),
        virtual.StaticVTabBuilder())
    extension_compiler.infer()
    extension_type = extension_compiler.compile()
    return extension_type
