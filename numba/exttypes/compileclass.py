# -*- coding: utf-8 -*-

import numba
from numba import error
from numba import typesystem
from numba import pipeline
from numba import symtab
from numba.minivect import minitypes

from numba.exttypes import logger
from numba.exttypes import signatures
from numba.exttypes import utils
from numba.exttypes import extension_types

from numba.typesystem.exttypes import methods
from numba.typesystem.exttypes import ordering
from numba.typesystem.exttypes import vtabtype
from numba.typesystem.exttypes import attributestype

class ExtensionCompiler(object):

    # [validators.MethodValidator]
    method_validators = None

    # [validators.ExtTypeValidator]
    exttype_validators = None

    def __init__(self, env, py_class, ext_type, flags,
                 method_maker, inheriter, attrbuilder, vtabbuilder):
        self.env = env
        self.py_class = py_class
        self.class_dict = dict(vars(py_class))
        self.ext_type = ext_type
        self.flags = flags

        self.inheriter = inheriter
        self.attrbuilder = attrbuilder
        self.vtabbuilder = vtabbuilder
        self.method_maker = method_maker

        # Partial function environments held after type inference has run
        self.func_envs = {}

    #------------------------------------------------------------------------
    # Type Inference
    #------------------------------------------------------------------------

    def infer(self):
        """
        Infer types:

            1) Inherit attributes and methods

                * Also build the vtab and attribute table types

            2) Process class attribute types:

                class Foo(object):
                    myattr = double

            3) Process method signatures @void(double) etc
            4) Infer extension attribute types from the __init__ method
        """
        self.inheriter.inherit(self.ext_type)
        process_class_attribute_types(self.ext_type, self.class_dict)

        # Process method signatures and set self.methods to [Method]
        self.methods = self.process_method_signatures()

        self.type_infer_init_method()
        self.type_infer_methods()

    def process_method_signatures(self):
        """
        Process all method signatures:

            * Verify signatures
            * Populate ext_type with method signatures (ExtMethodType)
        """
        processor = signatures.MethodSignatureProcessor(self.class_dict,
                                                        self.ext_type,
                                                        self.method_maker,
                                                        self.method_validators)

        methods, method_types = processor.get_method_signatures()

        # Update ext_type and class dict with known Method objects
        for method, method_type in zip(methods, method_types):
            self.ext_type.add_method(method)
            self.class_dict[method.name] = method

        return methods

    def type_infer_method(self, method):
        func_env = pipeline.compile2(self.env, method.py_func,
                                     method.signature.return_type,
                                     method.signature.args,
                                     pipeline_name='type_infer',
                                     **self.flags)
        self.func_envs[method] = func_env

        # Verify signature after type inference with registered
        # (user-declared) signature
        method.signature = methods.ExtMethodType(
            method.signature.return_type,
            method.signature.args,
            method.name,
            is_class=method.signature.is_class,
            is_static=method.signature.is_static)

        self.ext_type.add_method(method)

    def type_infer_init_method(self):
        initfunc = self.class_dict.get('__init__', None)
        if initfunc is None:
            return

        self.type_infer_method(initfunc)

        # Merge ext_type.symtab into attribute table
        table = self.ext_type.attribute_table
        for attr_name, variable in self.ext_type.symtab.iteritems():
            if attr_name not in table.attributedict:
                table.attributedict[attr_name] = variable.type

    def type_infer_methods(self):
        for method in self.methods:
            if method.name in ('__new__', '__init__'):
                continue

            self.type_infer_method(method)

    #------------------------------------------------------------------------
    # Finalize Tables
    #------------------------------------------------------------------------

    def finalize_tables(self):
        """
        Finalize (fix) the attribute and method tables.
        """
        self.attrbuilder.finalize(self.ext_type)
        self.vtabbuilder.finalize(self.ext_type)

    #------------------------------------------------------------------------
    # Validate
    #------------------------------------------------------------------------

    def validate(self):
        """
        Validate that we can build the extension type.
        """
        for validator in self.exttype_validators:
            validator.validate(self.ext_type)

    #------------------------------------------------------------------------
    # Compilation
    #------------------------------------------------------------------------

    def compile(self):
        """
        Compile extension methods:

            1) Process signatures such as @void(double)
            2) Infer native attributes through type inference on __init__
            3) Patch the extension type with a native attributes struct
            4) Infer types for all other methods
            5) Update the ext_type with a vtab type
            6) Compile all methods
        """
        self.class_dict['__numba_py_class'] = self.py_class
        self.compile_methods()
        vtable = self.vtabbuilder.build_vtab(self.ext_type)
        return self.build_extension_type(vtable)

    def compile_methods(self):
        """
        Compile all methods, reuse function environments from type inference
        stage.

        ∀ methods M sets M.lfunc and M.lfunc_pointer
        """

    def build_extension_type(self, vtable):
        """
        Build extension type from llvm methods and pointers and a populated
        virtual method table.
        """
        extension_type = extension_types.create_new_extension_type(
            self.py_class.__name__, self.py_class.__bases__, self.class_dict,
            self.ext_type, vtable, self.ext_type.vtab_type,
            self.ext_type.vtab_type.llvm_methods,
            self.ext_type.vtab_type.method_pointers)

        return extension_type


#------------------------------------------------------------------------
# Attribute Inheritance
#------------------------------------------------------------------------

class AttributesInheriter(object):
    """
    Inherit attributes and methods from parent classes:

        For attributes and methods ...

            1) Build a table type
            2) Copy supertype slots into subclass table type
    """

    def inherit(self, ext_type):
        "Inherit attributes and methods from superclasses"
        attr_table = self.build_attribute_table(ext_type)
        ext_type.attribute_table = attr_table

        vtable = self.build_method_table(ext_type)
        ext_type.vtab_type = vtable

    def build_attribute_table(self, ext_type):
        bases = utils.get_numba_bases(ext_type.py_class)

        parent_attrtables = [base.exttype.attribute_table for base in bases]

        attr_table = attributestype.ExtensionAttributesTableType(
            ext_type.py_class, parent_attrtables)

        for base in bases:
            self.inherit_attributes(attr_table, base.ext_type)

        return attr_table

    def build_method_table(self, ext_type):
        bases = utils.get_numba_bases(ext_type.py_class)

        parent_vtables = [base.exttype.vtab_type for base in bases]
        vtable = vtabtype.VTabType(ext_type.py_class, parent_vtables)

        for base in bases:
            self.inherit_methods(vtable, base.ext_type)

        return vtable

    def inherit_attributes(self, derived_ext_type, base_ext_type):
        """
        Inherit attributes from a parent class.
        May be called multiple times for multiple bases.
        """
        derived_ext_type.attribute_table.attributedict.update(
                    base_ext_type.attribute_table.attributedict)

    def inherit_methods(self, derived_ext_type, base_ext_type):
        """
        Inherit methods from a parent class.
        May be called multiple times for multiple bases.
        """
        derived_ext_type.vtab_type.methoddict.update(
                    base_ext_type.vtab_type.methoddict)


def process_class_attribute_types(ext_type, class_dict):
    """
    Process class attribute types:

        @jit
        class Foo(object):

            attr = double
    """
    for name, value in class_dict.iteritems():
        if isinstance(value, minitypes.Type):
            ext_type.symtab[name] = symtab.Variable(
                        value, promotable_type=False)

#------------------------------------------------------------------------
# Build Attributes
#------------------------------------------------------------------------

class AttributeBuilder(object):
    """
    Build attribute descriptors for Python-level access.
    """

    def finalize(self, ext_type):
        "Finalize the attribute table (and fix the order if necessary)"

    def create_descr(self, attr_name):
        """
        Create a descriptor that accesses the attribute from Python space.
        """

    def build_descriptors(self, env, py_class, ext_type, class_dict):
        "Cram descriptors into the class dict"
        for attr_name, attr_type in ext_type.symtab.iteritems():
            descriptor = self.create_descr(attr_name)
            class_dict[attr_name] = descriptor

#------------------------------------------------------------------------
# Build Virtual Method Table
#------------------------------------------------------------------------

class VTabBuilder(object):
    """
    Build virtual method table for quick calling from Numba.
    """

    def finalize(self, ext_type):
        "Finalize the method table (and fix the order if necessary)"

    def build_vtab(self, ext_type, method_pointers):
        """
        Build a virtual method table.
        The result will be kept alive on the extension type.
        """
