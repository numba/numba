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
        self.inheriter.inherit(
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

#------------------------------------------------------------------------
# Build Attributes
#------------------------------------------------------------------------

class AttributeBuilder(object):

    def build_attributes(self, ext_type):
        """
        Create attribute struct type from symbol table.
        """
        attrs = dict((name, var.type)
                         for name, var in ext_type.symtab.iteritems())

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

    def create_descr(self, attr_name):
        """
        Create a descriptor that accesses the attribute from Python space.
        """

    def build_descriptors(self, env, py_class, ext_type, class_dict):
        "Cram descriptors into the class dict"
        for attr_name, attr_type in ext_type.symtab.iteritems():
            descriptor = self.create_descr(attr_name)
            class_dict[attr_name] = descriptor