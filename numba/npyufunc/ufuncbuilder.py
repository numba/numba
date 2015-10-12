# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import warnings
import inspect
import numpy as np

from numba.decorators import jit
from numba.targets.registry import target_registry
from numba.targets.options import TargetOptions
from numba import utils, compiler, types, sigutils
from numba.numpy_support import as_dtype
from . import _internal
from .sigparse import parse_signature
from .wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.targets import registry


import llvmlite.llvmpy.core as lc

class UFuncTargetOptions(TargetOptions):
    OPTIONS = {
        "nopython" : bool,
        "forceobj" : bool,
    }


class UFuncTarget(registry.CPUTarget):
    options = UFuncTargetOptions


class UFuncDispatcher(object):
    targetdescr = UFuncTarget()

    def __init__(self, py_func, locals={}, targetoptions={}):
        self.py_func = py_func
        self.overloads = utils.UniqueDict()
        self.targetoptions = targetoptions
        self.locals = locals

    def compile(self, sig, locals={}, **targetoptions):
        locs = self.locals.copy()
        locs.update(locals)

        topt = self.targetoptions.copy()
        topt.update(targetoptions)

        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, topt)
        flags.set("no_compile")
        flags.set("no_cpython_wrapper")
        flags.set("error_model", "numpy")
        # Disable loop lifting
        # The feature requires a real python function
        flags.unset("enable_looplift")

        typingctx = self.targetdescr.typing_context
        targetctx = self.targetdescr.target_context

        args, return_type = sigutils.normalize_signature(sig)

        cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                      args=args, return_type=return_type,
                                      flags=flags, locals=locals)

        self.overloads[cres.signature] = cres
        return cres


target_registry['npyufunc'] = UFuncDispatcher

# Utility functions

def _compile_element_wise_function(nb_func, targetoptions, sig=None,
                                   argtypes=None, restype=None):
    # Handle argtypes
    if argtypes is not None:
        warnings.warn("Keyword argument argtypes is deprecated",
                      DeprecationWarning)
        assert sig is None
        if restype is None:
            sig = tuple(argtypes)
        else:
            sig = restype(*argtypes)

    # Do compilation
    # Return CompileResult to test
    cres = nb_func.compile(sig, **targetoptions)
    args, return_type = sigutils.normalize_signature(sig)

    return cres, args, return_type

def _finalize_ufunc_signature(cres, args, return_type):
    '''Given a compilation result, argument types, and a return type,
    build a valid Numba signature after validating that it doesn't
    violate the constraints for the compilation mode.
    '''
    if return_type is None:
        if cres.objectmode:
            # Object mode is used and return type is not specified
            raise TypeError("return type must be specified for object mode")
        else:
            return_type = cres.signature.return_type

    assert return_type != types.pyobject
    return return_type(*args)

def _build_element_wise_ufunc_wrapper(cres, signature):
    '''Build a wrapper for the ufunc loop entry point given by the
    compilation result object, using the element-wise signature.
    '''
    ctx = cres.target_context
    library = cres.library
    llvm_func = library.get_function(cres.fndesc.llvm_func_name)

    env = None
    if cres.objectmode:
        # Get env
        env = cres.environment
        assert env is not None
        ll_intp = cres.target_context.get_value_type(types.intp)
        ll_pyobj = cres.target_context.get_value_type(types.pyobject)
        envptr = lc.Constant.int(ll_intp, id(env)).inttoptr(ll_pyobj)
    else:
        envptr = None

    wrapper = build_ufunc_wrapper(library, ctx, llvm_func, signature,
                                  cres.objectmode, envptr, env)
    ptr = library.get_pointer_to_function(wrapper.name)

    # Get dtypes
    dtypenums = [as_dtype(a).num for a in signature.args]
    dtypenums.append(as_dtype(signature.return_type).num)
    return dtypenums, ptr, env

# Class definitions

class _BaseUFuncBuilder(object):

    _identities = {
        0: _internal.PyUFunc_Zero,
        1: _internal.PyUFunc_One,
        None: _internal.PyUFunc_None,
        }
    if np.__version__ >= '1.7':
        _identities["reorderable"] = _internal.PyUFunc_ReorderableNone

    @classmethod
    def parse_identity(cls, identity):
        try:
            identity = cls._identities[identity]
        except KeyError:
            raise ValueError("Invalid identity value %r" % (identity,))
        return identity

    def add(self, sig=None, argtypes=None, restype=None):
        if hasattr(self, 'targetoptions'):
            targetoptions = self.targetoptions
        else:
            targetoptions = self.nb_func.targetoptions
        cres, args, return_type = _compile_element_wise_function(
            self.nb_func, targetoptions, sig, argtypes, restype)
        sig = self._finalize_signature(cres, args, return_type)
        self._sigs.append(sig)
        self._cres[sig] = cres
        return cres

class UFuncBuilder(_BaseUFuncBuilder):

    def __init__(self, py_func, identity=None, targetoptions={}):
        self.py_func = py_func
        self.identity = self.parse_identity(identity)
        self.nb_func = jit(target='npyufunc', **targetoptions)(py_func)
        self._sigs = []
        self._cres = {}

    def _finalize_signature(self, cres, args, return_type):
        '''Slated for deprecation, use ufuncbuilder._finalize_ufunc_signature()
        instead.
        '''
        return _finalize_ufunc_signature(cres, args, return_type)

    def build_ufunc(self):
        dtypelist = []
        ptrlist = []
        if not self.nb_func:
            raise TypeError("No definition")

        # Get signature in the order they are added
        keepalive = []
        cres = None
        for sig in self._sigs:
            cres = self._cres[sig]
            dtypenums, ptr, env = self.build(cres, sig)
            dtypelist.append(dtypenums)
            ptrlist.append(utils.longint(ptr))
            keepalive.append((cres.library, env))

        datlist = [None] * len(ptrlist)

        if cres is None:
            argspec = inspect.getargspec(self.py_func)
            inct = len(argspec.args)
        else:
            inct = len(cres.signature.args)
        outct = 1

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        ufunc = _internal.fromfunc(self.py_func.__name__, self.py_func.__doc__,
                                   ptrlist, dtypelist, inct, outct, datlist,
                                   keepalive, self.identity)

        return ufunc

    def build(self, cres, signature):
        '''Slated for deprecation, use
        ufuncbuilder._build_element_wise_ufunc_wrapper().
        '''
        return _build_element_wise_ufunc_wrapper(cres, signature)


class GUFuncBuilder(_BaseUFuncBuilder):

    # TODO handle scalar
    def __init__(self, py_func, signature, identity=None, targetoptions={}):
        self.py_func = py_func
        self.identity = self.parse_identity(identity)
        self.nb_func = jit(target='npyufunc')(py_func)
        self.signature = signature
        self.sin, self.sout = parse_signature(signature)
        self.targetoptions = targetoptions
        self._sigs = []
        self._cres = {}

    def _finalize_signature(self, cres, args, return_type):
        if not cres.objectmode and cres.signature.return_type != types.void:
            raise TypeError("gufunc kernel must have void return type")

        if return_type is None:
            return_type = types.void

        return return_type(*args)

    def build_ufunc(self):
        dtypelist = []
        ptrlist = []
        if not self.nb_func:
            raise TypeError("No definition")

        # Get signature in the order they are added
        keepalive = []
        for sig in self._sigs:
            cres = self._cres[sig]
            dtypenums, ptr, env = self.build(cres)
            dtypelist.append(dtypenums)
            ptrlist.append(utils.longint(ptr))
            keepalive.append((cres.library, env))

        datlist = [None] * len(ptrlist)

        inct = len(self.sin)
        outct = len(self.sout)

        # Pass envs to fromfuncsig to bind to the lifetime of the ufunc object
        ufunc = _internal.fromfunc(self.py_func.__name__, self.py_func.__doc__,
                                   ptrlist, dtypelist, inct, outct, datlist,
                                   keepalive, self.identity, self.signature)
        return ufunc

    def build(self, cres):
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """
        # Buider wrapper for ufunc entry point
        ctx = cres.target_context
        library = cres.library
        signature = cres.signature
        llvm_func = library.get_function(cres.fndesc.llvm_func_name)
        wrapper, env = build_gufunc_wrapper(library, ctx, llvm_func,
                                            signature, self.sin, self.sout,
                                            fndesc=cres.fndesc,
                                            env=cres.environment)

        ptr = library.get_pointer_to_function(wrapper.name)

        # Get dtypes
        dtypenums = []
        for a in signature.args:
            if isinstance(a, types.Array):
                ty = a.dtype
            else:
                ty = a
            dtypenums.append(as_dtype(ty).num)
        return dtypenums, ptr, env

