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

import copy
from functools import partial

from numba import pipeline
from numba import typesystem
from numba import numbawrapper

from numba.exttypes import types as etypes
from numba.exttypes import virtual
from numba.exttypes import signatures
from numba.exttypes import validators
from numba.exttypes import compileclass
from numba.exttypes import ordering


#------------------------------------------------------------------------
# Build Attributes Struct
#------------------------------------------------------------------------

class AutojitAttributeBuilder(compileclass.AttributeBuilder):

    def finalize(self, ext_type):
        # TODO: hash-based attributes
        ext_type.attribute_table.create_attribute_ordering(ordering.extending)

    def create_descr(self, attr_name):
        """
        Create a descriptor that accesses the attribute on the ctypes struct.
        TODO: Use a perfect-hashed attribute table.
        """
        def _get(self):
            return getattr(self._numba_attrs, attr_name)
        def _set(self, value):
            return setattr(self._numba_attrs, attr_name, value)
        return property(_get, _set)

#------------------------------------------------------------------------
# Filter Methods
#------------------------------------------------------------------------

class AutojitMethodFilter(compileclass.Filterer):

    def filter(self, methods, ext_type):
        typed_methods = []

        for method in methods:
            if method.signature is None:
                # autojit method
                ext_type.vtab_type.untyped_methods[method.name] = method
            else:
                # method with signature
                typed_methods.append(method)

        return typed_methods

#------------------------------------------------------------------------
# Build Method Wrappers
#------------------------------------------------------------------------

class AutojitMethodWrapperBuilder(compileclass.MethodWrapperBuilder):

    def build_method_wrappers(self, env, extclass, ext_type):
        """
        Update the extension class with the function wrappers.
        """
        self.process_typed_methods(env, extclass, ext_type)
        self.process_untyped_methods(env, extclass, ext_type)

    def process_untyped_methods(self, env, extclass, ext_type):
        """
        Process autojit methods (undecorated methods). Use the fast
        NumbaSpecializingWrapper cache when for when we're being called
        from python. When we need to add a new specialization,
        `autojit_method_compiler` is invoked to compile the method.

        extclass: the extension type
        ext_type.py_class: the unspecialized class that was decorated
        """
        from numba.wrapping import compiler

        for method_name, method in ext_type.vtab_type.untyped_methods.iteritems():
            env.specializations.register(method.py_func)
            cache = env.specializations.get_autojit_cache(method.py_func)

            compiler_impl = compiler.MethodCompiler(env, extclass, method)
            wrapper = numbawrapper.NumbaSpecializingWrapper(
                method.py_func, compiler_impl, cache)

            setattr(extclass, method_name, wrapper)

# ______________________________________________________________________
# Compile method when called from Python

def autojit_method_compiler(env, extclass, method, signature):
    """
    Called to compile a new specialized method. The result should be
    added to the perfect hash-based vtable.
    """
    # compiled_method = numba.jit(argtypes=argtypes)(method.py_func)
    func_env = pipeline.compile2(env, method.py_func,
                                 restype=signature.return_type,
                                 argtypes=signature.args)

    # Create Method for the specialization
    new_method = signatures.Method(
        method.py_func,
        method.name,
        func_env.func_signature,
        is_class=method.is_class,
        is_static=method.is_static)

    new_method.update_from_env(func_env)

    # Update vtable type
    vtable_wrapper = extclass.__numba_vtab
    vtable_type = extclass.exttype.vtab_type
    vtable_type.specialized_methods[new_method.name,
                                    signature.args] = new_method

    # Replace vtable (which will update the vtable all (live) objects use)
    new_vtable = virtual.build_hashing_vtab(vtable_type)
    vtable_wrapper.replace_vtable(new_vtable)

    return func_env.numba_wrapper_func

#------------------------------------------------------------------------
# Autojit Extension Class Compiler
#------------------------------------------------------------------------

class AutojitExtensionCompiler(compileclass.ExtensionCompiler):
    """
    Compile @autojit extension classes.
    """

    method_validators = validators.autojit_validators
    exttype_validators = validators.autojit_type_validators

    def get_bases(self):
        # TODO: subclassing
        return (self.py_class,)

#------------------------------------------------------------------------
# Build Extension Type
#------------------------------------------------------------------------

def create_extension_compiler(env, py_class, flags, argtypes):
    """
    Create a partial environment to compile specialized versions of the
    extension class in.
    """
    from extensibletype import intern
    intern.global_intern_initialize()

    flags.pop('llvm_module', None)

    ext_type = etypes.AutojitExtensionType(py_class)

    extension_compiler = AutojitExtensionCompiler(
        env, py_class, ext_type, flags,
        signatures.AutojitMethodMaker(ext_type, argtypes),
        compileclass.AttributesInheriter(),
        AutojitMethodFilter(),
        AutojitAttributeBuilder(),
        virtual.HashBasedVTabBuilder(),
        AutojitMethodWrapperBuilder())

    extension_compiler.init()
    extension_compiler.infer()
    extension_compiler.finalize_tables()
    extension_compiler.validate()
    extension_type = extension_compiler.compile()

    return extension_type
