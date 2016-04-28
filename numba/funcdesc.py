"""
Function descriptors.
"""
from __future__ import print_function, division, absolute_import

from collections import defaultdict
import itertools
import sys
from types import ModuleType

from . import six, types


def transform_arg_name(arg):
    if isinstance(arg, types.Record):
        return "Record_%s" % arg._code
    elif (isinstance(arg, types.Array) and
          isinstance(arg.dtype, types.Record)):
        type_name = "array" if arg.mutable else "readonly array"
        return ("%s(Record_%s, %sd, %s)"
                % (type_name, arg.dtype._code, arg.ndim, arg.layout))
    else:
        return str(arg)


def default_mangler(name, argtypes):
    codedargs = '.'.join(transform_arg_name(a).replace(' ', '_')
                             for a in argtypes)
    return '.'.join([name, codedargs])


# A dummy module for dynamically-generated functions
_dynamic_modname = '<dynamic>'
_dynamic_module = ModuleType(_dynamic_modname)
_dynamic_module.__builtins__ = six.moves.builtins


class FunctionDescriptor(object):
    """
    Base class for function descriptors: an object used to carry
    useful metadata about a natively callable function.
    """
    __slots__ = ('native', 'modname', 'qualname', 'doc', 'typemap',
                 'calltypes', 'args', 'kws', 'restype', 'argtypes',
                 'mangled_name', 'unique_name', 'inline')

    _unique_ids = itertools.count(1)

    def __init__(self, native, modname, qualname, unique_name, doc,
                 typemap, restype, calltypes, args, kws, mangler=None,
                 argtypes=None, inline=False):
        self.native = native
        self.modname = modname
        self.qualname = qualname
        self.unique_name = unique_name
        self.doc = doc
        # XXX typemap and calltypes should be on the compile result,
        # not the FunctionDescriptor
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        if argtypes is not None:
            assert isinstance(argtypes, tuple), argtypes
            self.argtypes = argtypes
        else:
            # Get argument types from the type inference result
            # (note the "arg.FOO" convention as used in typeinfer
            self.argtypes = tuple(self.typemap['arg.' + a] for a in args)
        mangler = default_mangler if mangler is None else mangler
        # The mangled name *must* be unique, else the wrong function can
        # be chosen at link time.
        if self.modname:
            # XXX choose a different convention for object mode
            self.mangled_name = mangler('%s.%s' % (self.modname, self.unique_name),
                                        self.argtypes)
        else:
            self.mangled_name = mangler(self.unique_name, self.argtypes)
        self.inline = inline

    def lookup_module(self):
        """
        Return the module in which this function is supposed to exist.
        This may be a dummy module if the function was dynamically
        generated.
        """
        if self.modname == _dynamic_modname:
            return _dynamic_module
        else:
            return sys.modules[self.modname]

    def lookup_function(self):
        """
        Return the original function object described by this object.
        """
        return getattr(self.lookup_module(), self.qualname)

    @property
    def llvm_func_name(self):
        """
        The LLVM-registered name for the raw function.
        """
        return self.mangled_name

    # XXX refactor this

    @property
    def llvm_cpython_wrapper_name(self):
        """
        The LLVM-registered name for a CPython-compatible wrapper of the
        raw function (i.e. a PyCFunctionWithKeywords).
        """
        return 'cpython.' + self.mangled_name

    @property
    def llvm_cfunc_wrapper_name(self):
        """
        The LLVM-registered name for a C-compatible wrapper of the
        raw function.
        """
        return 'cfunc.' + self.mangled_name

    def __repr__(self):
        return "<function descriptor %r>" % (self.unique_name)

    @classmethod
    def _get_function_info(cls, interp):
        """
        Returns
        -------
        qualname, unique_name, modname, doc, args, kws, globals

        ``unique_name`` must be a unique name.
        """
        func = interp.bytecode.func
        qualname = interp.bytecode.func_qualname
        modname = func.__module__
        doc = func.__doc__ or ''
        args = tuple(interp.arg_names)
        kws = ()        # TODO

        if modname is None:
            # Dynamically generated function.
            modname = _dynamic_modname

        # Even the same function definition can be compiled into
        # several different function objects with distinct closure
        # variables, so we make sure to disambiguish using an unique id.
        unique_name = "%s$%d" % (qualname, next(cls._unique_ids))

        return qualname, unique_name, modname, doc, args, kws

    @classmethod
    def _from_python_function(cls, interp, typemap, restype, calltypes,
                              native, mangler=None, inline=False):
        (qualname, unique_name, modname, doc, args, kws,
         )= cls._get_function_info(interp)
        self = cls(native, modname, qualname, unique_name, doc,
                   typemap, restype, calltypes,
                   args, kws, mangler=mangler, inline=inline)
        return self


class PythonFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for Numba-compiled functions.
    """
    __slots__ = ()

    @classmethod
    def from_specialized_function(cls, interp, typemap, restype, calltypes,
                                  mangler, inline):
        """
        Build a FunctionDescriptor for a given specialization of a Python
        function (in nopython mode).
        """
        return cls._from_python_function(interp, typemap, restype, calltypes,
                                         native=True, mangler=mangler,
                                         inline=inline)

    @classmethod
    def from_object_mode_function(cls, interp):
        """
        Build a FunctionDescriptor for an object mode variant of a Python
        function.
        """
        typemap = defaultdict(lambda: types.pyobject)
        calltypes = typemap.copy()
        restype = types.pyobject
        return cls._from_python_function(interp, typemap, restype, calltypes,
                                         native=False)


class ExternalFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for opaque external functions
    (e.g. raw C functions).
    """
    __slots__ = ()

    def __init__(self, name, restype, argtypes):
        args = ["arg%d" % i for i in range(len(argtypes))]
        super(ExternalFunctionDescriptor, self).__init__(native=True,
                modname=None, qualname=name, unique_name=name, doc='',
                typemap=None, restype=restype, calltypes=None,
                args=args, kws=None, mangler=lambda a, x: a,
                argtypes=argtypes)
