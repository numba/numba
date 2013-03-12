"""
Compiling @autojit extension classes works as follows:

    * Create an extension Numba/minivect type holding a symtab
    * Capture attribute types in the symtab in the same was as @jit
    * Build attribute hash-based vtable, hashing on (attr_name, attr_type).

        (attr_name, attr_type) is the only allowed key for that attribute
        (i.e. this is fixed at compile time (for now). This means consumers
        will always know the attribute type (and don't need to specialize
        on different attribute types).

        However, using a hash-based attribute table allows easy implementation
        of multiple inheritance (virtual inheritance), without complicated
        C++ dynamic offsets to base objects (see also virtual.py).

    For all methods M with static input types:
        * Compile M
        * Register M in a list of compiled methods

    * Build initial hash-based virtual method table from compiled methods

        * Create pre-hash values for the signatures
            * We use these values to look up methods at runtime

        * Parametrize the virtual method table to build a final hash function:

            slot_index = (((prehash >> table.r) & self.table.m_f) ^
                           self.displacements[prehash & self.table.m_g])

            See also virtual.py and the following SEPs:

                https://github.com/numfocus/sep/blob/master/sep200.rst
                https://github.com/numfocus/sep/blob/master/sep201.rst

            And the following paper to understand the perfect hashing scheme:

                Hash and Displace: Efficient Evaluation of Minimal Perfect
                Hash Functions (1999) by Rasmus Pagn:

                    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.6530

    * Create descriptors that wrap the native attributes
    * Create an extension type:

            {
                hash-based virtual method table (PyCustomSlots_Table **)
                PyGC_HEAD
                PyObject_HEAD
                ...
                native attributes
            }

        We precede the object with the table to make this work in a more
        generic scheme, e.g. where a caller is dealing with an unknown
        object, and we quickly want to see whether it support such a
        perfect-hashing virtual method table:

            if (o->ob_type->tp_flags & NATIVELY_CALLABLE_TABLE) {
                PyCustomSlots_Table ***slot_p = ((char *) o) - sizeof(PyGC_HEAD)
                PyCustomSlots_Table *vtab = **slot_p
                look up function
            } else {
                PyObject_Call(...)
            }

        We need to store a PyCustomSlots_Table ** in the object to allow
        the producer of the table to replace the table with a new table
        for all live objects (e.g. by adding a specialization for
        an autojit method).
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
from numba.exttypes import validators
from numba.exttypes import compileclass
from numba.exttypes import extension_types


#------------------------------------------------------------------------
# Populate Extension Type with Methods
#------------------------------------------------------------------------

class AutojitExtensionCompiler(compileclass.ExtensionCompiler):
    """
    Compile @autojit extension classes.
    """

    method_validators = validators.autojit_validators
    exttype_validators = validators.autojit_type_validators

    def inherit_method(self, method_name, slot_idx):
        "Inherit a method from a superclass in the vtable"
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
            method_pointers.append((method_name, func_env.translator.lfunc_pointer))

            self.class_dict[method_name] = method.result(func_env.numba_wrapper_func)

        return method_pointers, lmethods

#------------------------------------------------------------------------
# Attribute Inheritance
#------------------------------------------------------------------------

class AutojitAttributesInheriter(compileclass.AttributesInheriter):
    """
    Inherit attributes and methods from parent classes.
    """

    def inherit_attributes(self, ext_type, parent_struct_type):
        ext_type.parent_attr_struct = parent_struct_type
        ext_type.attribute_struct = numba.struct(parent_struct_type.fields)

        for field_name, field_type in ext_type.attribute_struct.fields:
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

class AutojitAttributeBuilder(compileclass.AttributeBuilder):

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

    ext_type = typesystem.AutojitExtensionType(py_class)

    extension_compiler = AutojitExtensionCompiler(
        env, py_class, ext_type, flags,
        AutojitAttributesInheriter(),
        AutojitAttributeBuilder(),
        virtual.HashBasedVTabBuilder())

    extension_compiler.infer()
    extension_type = extension_compiler.compile()

    return extension_type