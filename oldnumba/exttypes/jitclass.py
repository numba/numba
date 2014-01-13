"""
Compiling @jit extension classes works as follows:

    * Create an extension Numba type holding a symtab
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

from numba import typesystem

from numba.exttypes import virtual
from numba.exttypes import signatures
from numba.exttypes import validators
from numba.exttypes import compileclass
from numba.exttypes import ordering
from numba.exttypes import types as etypes

#------------------------------------------------------------------------
# Jit Extension Class Compiler
#------------------------------------------------------------------------

class JitExtensionCompiler(compileclass.ExtensionCompiler):
    """
    Compile @jit extension classes.
    """

    method_validators = validators.jit_validators
    exttype_validators = validators.jit_type_validators

#------------------------------------------------------------------------
# Build Attributes Struct
#------------------------------------------------------------------------

class JitAttributeBuilder(compileclass.AttributeBuilder):

    def finalize(self, ext_type):
        ext_type.attribute_table.create_attribute_ordering(ordering.extending)

    def create_descr(self, attr_name):
        """
        Create a descriptor that accesses the attribute on the ctypes struct.
        This is set by the extension type constructor __new__.
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

    # ext_type = etypes.jit_exttype(py_class)
    ext_type = typesystem.jit_exttype(py_class)

    extension_compiler = JitExtensionCompiler(
        env, py_class, dict(vars(py_class)), ext_type, flags,
        signatures.JitMethodMaker(),
        compileclass.AttributesInheriter(),
        compileclass.Filterer(),
        JitAttributeBuilder(),
        virtual.StaticVTabBuilder(),
        compileclass.MethodWrapperBuilder())

    extension_compiler.init()
    extension_compiler.infer()
    extension_compiler.finalize_tables()
    extension_compiler.validate()
    extension_type = extension_compiler.compile()

    return extension_type
