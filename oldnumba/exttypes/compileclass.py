# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from numba import pipeline
from numba import symtab
from numba import typesystem

from numba.exttypes import signatures
from numba.exttypes import utils
from numba.exttypes import extension_types
from numba.exttypes import methodtable
from numba.exttypes import attributetable
from numba.exttypes.types import methods

class ExtensionCompiler(object):

    # [validators.MethodValidator]
    method_validators = None

    # [validators.ExtTypeValidator]
    exttype_validators = None

    def __init__(self, env, py_class, class_dict, ext_type, flags,
                 method_maker,
                 inheriter,
                 method_filter,
                 attrbuilder,
                 vtabbuilder,
                 methodwrapper):
        self.env = env
        self.py_class = py_class
        self.class_dict = class_dict
        self.ext_type = ext_type
        self.flags = flags

        self.inheriter = inheriter
        self.method_filter = method_filter
        self.attrbuilder = attrbuilder
        self.vtabbuilder = vtabbuilder
        self.method_maker = method_maker
        self.methodwrapper = methodwrapper

        # Partial function environments held after type inference has run
        self.func_envs = {}

    #------------------------------------------------------------------------
    # Initialized and Inheritance
    #------------------------------------------------------------------------

    def init(self):
        """
        Initialize:

            1) Inherit attributes and methods

                * Also build the vtab and attribute table types

            2) Process class attribute types:

                class Foo(object):
                    myattr = double

            3) Process method signatures @void(double) etc
        """
        self.class_dict['__numba_py_class'] = self.py_class

        self.inheriter.inherit(self.ext_type)
        process_class_attribute_types(self.ext_type, self.class_dict)

        # Process method signatures and set self.methods to [Method]
        self.methods = self.process_method_signatures()

        # Build ext_type.symtab
        build_extension_symtab(self.ext_type)

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

        methods = processor.get_method_signatures()
        methods = self.method_filter.filter(methods, self.ext_type)

        # Update ext_type and class dict with known Method objects
        for method in methods:
            self.ext_type.add_method(method)
            self.class_dict[method.name] = method

        return methods

    #------------------------------------------------------------------------
    # Type Inference
    #------------------------------------------------------------------------

    def infer(self):
        """
            1) Infer extension attribute types from the __init__ method
            2) Type infer all methods
        """
        # Update ext_type.symtab
        self.type_infer_init_method()

        # Type infer the rest of the methods (with fixed attribute table!)
        self.type_infer_methods()

    def type_infer_method(self, method):
        func_env = pipeline.compile2(self.env, method.py_func,
                                     method.signature.return_type,
                                     method.signature.args,
                                     # don't use qualified name
                                     name=method.name,
                                     pipeline_name='type_infer',
                                     **self.flags)
        self.func_envs[method] = func_env

        # Verify signature after type inference with registered
        # (user-declared) signature
        method.signature = methods.ExtMethodType(
            method.signature.return_type,
            method.signature.args,
            method.name,
            is_class_method=method.is_class,
            is_static_method=method.is_static)

        self.ext_type.add_method(method)

    def type_infer_init_method(self):
        initfunc = self.class_dict.get('__init__', None)
        if initfunc is None:
            return

        self.type_infer_method(initfunc)

    def type_infer_methods(self):
        for method in self.methods:
            if method.name not in ('__new__', '__init__') and method.signature:
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
        self.compile_methods()

        vtable = self.vtabbuilder.build_vtab(self.ext_type)
        extclass = self.build_extension_type(vtable)

        # Set the extension class on the type. We may instead want an
        # ExtensionEnvironment associated with each ext_type, but this
        # would be a global thing
        self.ext_type.extclass = extclass

        self.attrbuilder.build_descriptors(self.ext_type, extclass)
        self.methodwrapper.build_method_wrappers(
            self.env, extclass, self.ext_type)

        return extclass

    def compile_methods(self):
        """
        Compile all methods, reuse function environments from type inference
        stage.

        ∀ methods M sets M.lfunc, M.lfunc_pointer and M.wrapper_func
        """
        for i, method in enumerate(self.methods):
            func_env = self.func_envs[method]
            pipeline.run_env(self.env, func_env, pipeline_name='compile')
            method.update_from_env(func_env)

    def get_bases(self):
        """
        Get base classes for the resulting extension type.

        For jit types, these are simply the bases of the Python class we
        decorated. For autojit-decorated classes we get a more complicated
        inheritance scheme (see AutojitExtensionCompiler.get_bases).
        """
        return self.py_class.__bases__

    def get_metacls(self):
        """
        Return the metaclass for the specialized extension type.
        """
        return type

    def build_extension_type(self, vtable):
        """
        Build extension type from llvm methods and pointers and a populated
        virtual method table.
        """
        vtable_wrapper = self.vtabbuilder.wrap_vtable(vtable)

        extension_type = extension_types.create_new_extension_type(
            self.get_metacls(),
            self.py_class.__name__, self.get_bases(), self.class_dict,
            self.ext_type, vtable_wrapper)

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

        attr_table = attributetable.AttributeTable(
            ext_type.py_class, parent_attrtables)

        for base in bases:
            self.inherit_attributes(attr_table, base.exttype)

        return attr_table

    def build_method_table(self, ext_type):
        bases = utils.get_numba_bases(ext_type.py_class)

        parent_vtables = [base.exttype.vtab_type for base in bases]
        vtable = methodtable.VTabType(ext_type.py_class, parent_vtables)

        for base in bases:
            self.inherit_methods(vtable, base.exttype)

        return vtable

    def inherit_attributes(self, attr_table, base_ext_type):
        """
        Inherit attributes from a parent class.
        May be called multiple times for multiple bases.
        """
        base_attrs = base_ext_type.attribute_table.attributedict
        attr_table.inherited.update(base_attrs)     # { attr_name }
        attr_table.attributedict.update(base_attrs) # { attr_name : attr_type }

    def inherit_methods(self, vtable, base_ext_type):
        """
        Inherit methods from a parent class.
        May be called multiple times for multiple bases.
        """
        base_methods = base_ext_type.vtab_type.methoddict
        vtable.inherited.update(base_methods)   # { method_name }
        vtable.methoddict.update(base_methods)  # { method_name : Method }

#------------------------------------------------------------------------
# Extension Attribute Processing
#------------------------------------------------------------------------

def process_class_attribute_types(ext_type, class_dict):
    """
    Process class attribute types:

        @jit
        class Foo(object):

            attr = double
    """
    table = ext_type.attribute_table
    for name, value in class_dict.iteritems():
        if isinstance(value, typesystem.Type):
            table.attributedict[name] = value

def build_extension_symtab(ext_type):
    """
    Create symbol table for all attributes of the extension type. These
    are Variables which are used by the type inferencer and used to
    type check attribute assignments.

    New attribute assignments create new ExtensionAttributeVariable
    variables in the symtab. These variables update the attribute table
    during type inference:

        class Foo(object):

            value1 = double

            def __init__(self, value2):
                self.value2 = int_(value2)

    Before type inference of __init__ we have:

        symtab = { 'value1': Variable(double) }

    and after type inference of __init__ we have:

        symtab = {
            'value1': Variable(double),                   # type is fixed
            'value2': ExtensionAttributeVariable(int_),   # type is inferred
        }
    """
    table = ext_type.attribute_table
    for attr_name, attr_type in table.attributedict.iteritems():
        ext_type.symtab[attr_name] = symtab.Variable(attr_type,
                                                     promotable_type=False)

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

    def build_descriptors(self, ext_type, extension_class):
        "Cram descriptors into the class dict"
        table = ext_type.attribute_table

        for attr_name, attr_type in table.attributedict.iteritems():
            descriptor = self.create_descr(attr_name)
            setattr(extension_class, attr_name, descriptor)

#------------------------------------------------------------------------
# Build Method Wrappers
#------------------------------------------------------------------------

class MethodWrapperBuilder(object):

    def build_method_wrappers(self, env, extclass, ext_type):
        """
        Update the extension class with the function wrappers.
        """
        self.process_typed_methods(env, extclass, ext_type)

    def process_typed_methods(self, env, extclass, ext_type):
        for method in ext_type.methoddict.itervalues():
            setattr(extclass, method.name, method.get_wrapper())

#------------------------------------------------------------------------
# Filters
#------------------------------------------------------------------------

class Filterer(object):
    def filter(self, iterable, *args):
        return list(iterable)