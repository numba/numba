"""
Compiling @autojit extension classes works as follows:

    * Create an extension Numba type holding a symtab
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

from numba import error
from numba import pipeline
from numba import numbawrapper
from numba import typesystem

from numba.exttypes import types as etypes
from numba.exttypes import utils
from numba.exttypes import virtual
from numba.exttypes import signatures
from numba.exttypes import validators
from numba.exttypes import compileclass
from numba.exttypes import ordering
from numba.exttypes import autojitmeta


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
        """
        Get base classes for the resulting extension type.

        We can try several inheritance schemes. We can choose between:

            1) One unspecialized - general - class

                * This must bind specialized methods on each object instance
                  to allow method calls from Python, making object allocation
                  more expensive

                * We must take the max() of tp_basicsize to allow enough
                  space for all specializations. However, the specialization
                  universe is not known up front, so we must allocate
                  attributes separately on the heap (or perhaps we must
                  override tp_alloc to use a dynamic size depending on the
                  specialization - but this may conflict with non-numba base
                  classes).

            2) One specialized class per instance

                * This allows us to experiment with static layouts for
                  attributes or methods more easily

                * It seems more intuitive to back specialized objects with
                  a specialized type

        We will go with 2). We could go for a specialization tree as follows:

                     A
                   / | \
                A0   |  A1
                 |   |  |
                 |   B  |
                 | /  \ |
                B0     B1

        Which gets us:

            issubclass(A_specialized, A)
            isinstance(A_specialized(), A)

        as well as

            issubclass(B_specialized, A_specialized)

        However, to support this scheme, the unspecialized class A must:

            1) Be subclassable
            2) Return specialized object instances when instantiated
            3) Support unbound method calls

        1) requires that A be a class, and then 2) implies that A's metaclass
        overrides __call__ or that A implements __new__.

        However, since A_specialized subclasses A, A_specialized.__new__ would
        need to skip A.__new__, which requires numba to insert a __new__
        or modify a user's __new__ method in A_specialized.

        The metaclass option seems more feasible:

            A_meta.__call__ -> specialized object instance

        Users can then override a metaclass in a Python (or numba?) subclass
        as follows:

            class MyMeta(type(MyNumbaClass)):
                ...

        The metaclass can also support indexing:

            A_specialized = A[{'attrib_a': double}]
        """
        # TODO: subclassing
        return (self.py_class,)

    def get_metacls(self):
        return autojitmeta.create_specialized_metaclass(self.py_class)

#------------------------------------------------------------------------
# Unbound Methods from Python
#------------------------------------------------------------------------

class UnboundDelegatingMethod(object):
    """
    Function in the unspecialized class that is used for delegation to
    a method in a specialized class, i.e.

        A.method(A(10.0)) -> A(10.0).method()

    This method can never be bound, since __new__ always returns specialized
    instances (so the unspecialized class cannot be instantiated!).
    """

    def __init__(self, py_class, name):
        self.py_class = py_class
        self.name = name

    def __call__(self, obj, *args, **kwargs):
        numbawrapper.unbound_method_type_check(self.py_class, obj)
        return getattr(obj, self.name)(*args, **kwargs)

def make_delegations(py_class):
    """
    Make delegation unbound methods that delegate from the unspecialized
    class to the specialized class. E.g.

        m = A.method
        m(A(10.0))      # Delegate to A[double].method
    """
    class_dict = vars(py_class)
    for name, func in class_dict.iteritems():
        if isinstance(func, (typesystem.Function, staticmethod, classmethod)):
            method = signatures.process_signature(func, name)
            if method.is_class or method.is_static:
                # Class or static method: use the pure Python function wrapped
                # in classmethod()/staticmethod()
                method.wrapper_func = method.py_func
                new_func = method.get_wrapper()
            else:
                # Regular unbound method. Create dispatcher to bound method
                # when called
                new_func = UnboundDelegatingMethod(py_class, name)

            setattr(py_class, name, new_func)

        #------------------------------------------------------------------------
# Make Specializing Class -- Entry Point for decorator application
#------------------------------------------------------------------------

def autojit_class_wrapper(py_class, compiler_impl, cache):
    """
    Invoked when a class is decorated with @autojit.

    :param py_class: decorated python class
    :param compiler_impl: compiler.ClassCompiler
    :param cache: FunctionCache
    :return: py_class that returns specialized object instances
    """
    from numba import numbawrapper

    if utils.is_numba_class(py_class):
        raise error.NumbaError("Subclassing not yet supported "
                               "for autojit classes")

    # runtime_args -> specialized extension type instance
    class_specializer = numbawrapper.NumbaSpecializingWrapper(
        py_class, compiler_impl, cache)

    py_class = autojitmeta.create_unspecialized_cls(py_class, class_specializer)
    py_class.__is_numba_autojit = True

    # Update the class from which we derive specializations (which will be
    # passed to create_extension() below)
    compiler_impl.py_func = py_class

    # Back up class dict, since we're going to modify it
    py_class.__numba_class_dict = dict(vars(py_class))

    # Make delegation methods for unbound methods
    make_delegations(py_class)

    # Setup up partial compilation environment
    # partial_env = create_partial_extension_environment(
    #     compiler_impl.env, py_class, compiler_impl.flags)
    # compiler_impl.partial_ext_env = partial_env

    return py_class

#------------------------------------------------------------------------
# Build Extension Type -- Compiler Entry Point
#------------------------------------------------------------------------

# def create_partial_extension_environment(env, py_class, flags, argtypes):
def create_extension(env, py_class, flags, argtypes):
    """
    Create a partial environment to compile specialized versions of the
    extension class in.

    Inovked when calling the wrapped class to compile a specialized
    new extension type.
    """
    # TODO: Remove argtypes! Partial environment!

    flags.pop('llvm_module', None)

    ext_type = etypes.autojit_exttype(py_class)

    class_dict = dict(utils.get_class_dict(py_class))

    extension_compiler = AutojitExtensionCompiler(
        env, py_class, class_dict, ext_type, flags,
        signatures.AutojitMethodMaker(argtypes),
        compileclass.AttributesInheriter(),
        AutojitMethodFilter(),
        AutojitAttributeBuilder(),
        virtual.HashBasedVTabBuilder(),
        AutojitMethodWrapperBuilder())

    extension_compiler.init()
    # return extension_compiler

# def compile_class(extension_compiler, argtypes):
    extension_compiler.infer()
    extension_compiler.finalize_tables()
    extension_compiler.validate()
    extension_type = extension_compiler.compile()

    return extension_type