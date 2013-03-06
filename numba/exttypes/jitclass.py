"""
Compiling extension classes works as follows:

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

See also extension_types.pyx
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
from numba.exttypes import extension_types

#------------------------------------------------------------------------
# Populate Extension Type with Methods
#------------------------------------------------------------------------

class ExtensionCompiler(object):

    def __init__(self, env, py_class, ext_type, flags,
                 inheriter, attrbuilder, vtabbuilder):
        self.env = env
        self.py_class = py_class
        self.class_dict = dict(vars(py_class))
        self.ext_type = ext_type
        self.flags = flags

        self.inheriter = inheriter
        self.attrbuilder = attrbuilder
        self.vtabbuilder = vtabbuilder

        # Partial function environments held after type inference has run
        self.func_envs = {}

    #------------------------------------------------------------------------
    # Type Inference
    #------------------------------------------------------------------------

    def infer(self):
        self.infer_attributes()
        self.process_method_signatures()
        self.type_infer_init_method()
        self.attrbuilder.build_attributes(self.ext_type)
        self.type_infer_methods()
        self.vtabbuilder.build_vtab_type(self.ext_type)

    def infer_attributes(self):
        self.inheriter.inherit_attributes(
            self.ext_type, self.class_dict)
        self.inheriter.process_class_attribute_types(
            self.ext_type, self.class_dict)

    def process_method_signatures(self):
        """
        Process all method signatures:

            * Verify signatures
            * Populate ext_type with method signatures (ExtMethodType)
        """
        method_maker = signatures.JitMethodMaker(self.ext_type)
        processor = signatures.MethodSignatureProcessor(self.class_dict,
                                                        self.ext_type,
                                                        method_maker)

        for method, method_type in processor.get_method_signatures():
            self.ext_type.add_method(method.name, method_type)
            self.class_dict[method.name] = method

    def type_infer_method(self, method, method_name):
        if method_name not in self.ext_type.methoddict:
            return

        signature = self.ext_type.get_signature(method_name)
        restype, argtypes = signature.return_type, signature.args

        func_env = pipeline.compile2(self.env, method.py_func,
                                     restype, argtypes,
                                     pipeline_name='type_infer',
                                     **self.flags)
        self.func_envs[method] = func_env

        self.ext_type.add_method(method_name, func_env.func_signature)

    def type_infer_init_method(self):
        initfunc = self.class_dict.get('__init__', None)
        if initfunc is None:
            return

        self.type_infer_method(initfunc, '__init__')

    def type_infer_methods(self):
        for method_name, method in self.class_dict.iteritems():
            if method_name in ('__new__', '__init__') or method is None:
                continue

            self.type_infer_method(method, method_name)

    #------------------------------------------------------------------------
    # Compilation
    #------------------------------------------------------------------------

    def compile(self):
        """
        Compile extension methods:

            1) Process signatures such as @void(double)
            2) Infer native attributes through type inference on __init__
            3) Path the extension type with a native attributes struct
            4) Infer types for all other methods
            5) Update the ext_type with a vtab type
            6) Compile all methods
        """
        self.class_dict['__numba_py_class'] = self.py_class
        method_pointers, lmethods = self.compile_methods()
        vtab = self.vtabbuilder.build_vtab(self.ext_type, method_pointers)
        return self.build_extension_type(lmethods, method_pointers, vtab)

    def inherit_method(self, method_name, *args, **kwargs):
        """
        Inherit a method from a superclass in the vtable.

        :return: a pointer to the function.
        """

    def compile_methods(self):
        """
        Compile all methods, reuse function environments from type inference
        stage.

        :return: ([method_pointers], [llvm_funcs])
        """

    def build_extension_type(self, lmethods, method_pointers, vtab):
        """
        Build extension type from llvm methods and pointers and a populated
        virtual method table.
        """
        extension_type = extension_types.create_new_extension_type(
            self.py_class.__name__, self.py_class.__bases__, self.class_dict,
            self.ext_type, vtab, self.ext_type.vtab_type,
            lmethods, method_pointers)

        return extension_type


class JitExtensionCompiler(ExtensionCompiler):

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

class AttributesInheriter(object):

    def inherit(self, ext_type, class_dict):
        self.inherit_attributes(ext_type, class_dict)
        self.process_class_attribute_types(ext_type, class_dict)

    def inherit_attributes(self, ext_type, class_dict):
        "Inherit attributes and methods from superclasses"
        cls = ext_type.py_class
        if not is_numba_class(cls):
            # superclass is not a numba class
            return

        struct_type = utils.get_struct_type(cls)
        vtab_type = utils.get_vtab_type(cls)
        self._verify_base_class_compatibility(cls, struct_type, vtab_type)

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

    def process_class_attribute_types(self, ext_type, class_dict):
        """
        Process class attribute types:

            @jit
            class Foo(object):

                attr = double
        """
        for name, value in class_dict.iteritems():
            if isinstance(value, minitypes.Type):
                ext_type.symtab[name] = symtab.Variable(value,
                                                        promotable_type=False)

    def _verify_base_class_compatibility(self, py_class, struct_type, vtab_type):
        "Verify that we can build a compatible class layout"
        bases = [py_class]
        for base in py_class.__bases__:
            if is_numba_class(base):
                attr_prefix = utils.get_struct_type(base).is_prefix(struct_type)
                method_prefix = utils.get_vtab_type(base).is_prefix(vtab_type)
                if not attr_prefix or not method_prefix:
                    raise error.NumbaError(
                                "Multiple incompatible base classes found: "
                                "%s and %s" % (base, bases[-1]))

                bases.append(base)

#------------------------------------------------------------------------
# Build Attributes Struct
#------------------------------------------------------------------------

class AttributeBuilder(object):

    def build_attributes(self, ext_type):
        """
        Create attribute struct type from symbol table.
        """
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

    def _create_descr(self, attr_name):
        """
        Create a descriptor that accesses the attribute on the ctypes struct.
        """
        def _get(self):
            return getattr(self._numba_attrs, attr_name)
        def _set(self, value):
            return setattr(self._numba_attrs, attr_name, value)
        return property(_get, _set)

    def build_descriptors(self, env, py_class, ext_type, class_dict):
        "Cram descriptors into the class dict"
        for attr_name, attr_type in ext_type.symtab.iteritems():
            descriptor = self._create_descr(attr_name)
            class_dict[attr_name] = descriptor


#------------------------------------------------------------------------
# Build Extension Type
#------------------------------------------------------------------------

def create_extension(env, py_class, flags):
    """
    Compile an extension class given the NumbaEnvironment and the Python
    class that contains the functions that are to be compiled.
    """
    flags.pop('llvm_module', None)

    ext_type = typesystem.ExtensionType(py_class)

    extension_compiler = JitExtensionCompiler(env, py_class, ext_type, flags,
                                              AttributesInheriter(),
                                              AttributeBuilder(),
                                              virtual.StaticVTabBuilder())
    extension_compiler.infer()
    extension_type = extension_compiler.compile()
    return extension_type
