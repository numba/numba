# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import inspect
from contextlib import contextmanager

from numba import serialize
from numba.decorators import jit
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from numba.targets.registry import dispatcher_registry, cpu_target
from numba.targets.cpu import FastMathOptions
from numba import utils, compiler, types, sigutils
from numba.numpy_support import as_dtype
from . import _internal
from .sigparse import parse_signature
from .wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.caching import FunctionCache, NullCache
from numba.compiler_lock import global_compiler_lock
from numba.config import PYVERSION


class UFuncTargetOptions(TargetOptions):
    OPTIONS = {
        "nopython" : bool,
        "forceobj" : bool,
        "boundscheck": bool,
        "fastmath" : FastMathOptions,
    }


class UFuncTarget(TargetDescriptor):
    options = UFuncTargetOptions

    @property
    def typing_context(self):
        return cpu_target.typing_context

    @property
    def target_context(self):
        return cpu_target.target_context


ufunc_target = UFuncTarget()


class UFuncDispatcher(object):
    """
    An object handling compilation of various signatures for a ufunc.
    """
    targetdescr = ufunc_target

    def __init__(self, py_func, locals={}, targetoptions={}):
        self.py_func = py_func
        self.overloads = utils.UniqueDict()
        self.targetoptions = targetoptions
        self.locals = locals
        self.cache = NullCache()

    def __reduce__(self):
        globs = serialize._get_function_globals_for_reduction(self.py_func)
        return (
            serialize._rebuild_reduction,
            (
                self.__class__,
                serialize._reduce_function(self.py_func, globs),
                self.locals, self.targetoptions,
            ),
        )

    @classmethod
    def _rebuild(cls, redfun, locals, targetoptions):
        return cls(serialize._rebuild_function(*redfun),
                   locals, targetoptions)

    def enable_caching(self):
        self.cache = FunctionCache(self.py_func)

    def compile(self, sig, locals={}, **targetoptions):
        locs = self.locals.copy()
        locs.update(locals)

        topt = self.targetoptions.copy()
        topt.update(targetoptions)

        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, topt)

        flags.set("no_cpython_wrapper")
        flags.set("error_model", "numpy")
        # Disable loop lifting
        # The feature requires a real python function
        flags.unset("enable_looplift")

        return self._compile_core(sig, flags, locals)

    def _compile_core(self, sig, flags, locals):
        """
        Trigger the compiler on the core function or load a previously
        compiled version from the cache.  Returns the CompileResult.
        """
        typingctx = self.targetdescr.typing_context
        targetctx = self.targetdescr.target_context

        @contextmanager
        def store_overloads_on_success():
            # use to ensure overloads are stored on success
            try:
                yield
            except Exception:
                raise
            else:
                exists = self.overloads.get(cres.signature)
                if exists is None:
                    self.overloads[cres.signature] = cres

        # Use cache and compiler in a critical section
        with global_compiler_lock:
            with store_overloads_on_success():
                # attempt look up of existing
                cres = self.cache.load_overload(sig, targetctx)
                if cres is not None:
                    return cres

                # Compile
                args, return_type = sigutils.normalize_signature(sig)
                cres = compiler.compile_extra(typingctx, targetctx,
                                              self.py_func, args=args,
                                              return_type=return_type,
                                              flags=flags, locals=locals)

                # cache lookup failed before so safe to save
                self.cache.save_overload(sig, cres)

                return cres


dispatcher_registry['npyufunc'] = UFuncDispatcher


# Utility functions

def _compile_element_wise_function(nb_func, targetoptions, sig):
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
    fname = cres.fndesc.llvm_func_name

    with global_compiler_lock:
        info = build_ufunc_wrapper(library, ctx, fname, signature,
                                   cres.objectmode, cres)
        ptr = info.library.get_pointer_to_function(info.name)
    # Get dtypes
    dtypenums = [as_dtype(a).num for a in signature.args]
    dtypenums.append(as_dtype(signature.return_type).num)
    return dtypenums, ptr, cres.environment


_identities = {
    0: _internal.PyUFunc_Zero,
    1: _internal.PyUFunc_One,
    None: _internal.PyUFunc_None,
    "reorderable": _internal.PyUFunc_ReorderableNone,
}


def parse_identity(identity):
    """
    Parse an identity value and return the corresponding low-level value
    for Numpy.
    """
    try:
        identity = _identities[identity]
    except KeyError:
        raise ValueError("Invalid identity value %r" % (identity,))
    return identity


# Class definitions

class _BaseUFuncBuilder(object):

    def add(self, sig=None):
        if hasattr(self, 'targetoptions'):
            targetoptions = self.targetoptions
        else:
            targetoptions = self.nb_func.targetoptions
        cres, args, return_type = _compile_element_wise_function(
            self.nb_func, targetoptions, sig)
        sig = self._finalize_signature(cres, args, return_type)
        self._sigs.append(sig)
        self._cres[sig] = cres
        return cres

    def disable_compile(self):
        """
        Disable the compilation of new signatures at call time.
        """
        # Override this for implementations that support lazy compilation


class UFuncBuilder(_BaseUFuncBuilder):

    def __init__(self, py_func, identity=None, cache=False, targetoptions={}):
        self.py_func = py_func
        self.identity = parse_identity(identity)
        self.nb_func = jit(target='npyufunc',
                           cache=cache,
                           **targetoptions)(py_func)
        self._sigs = []
        self._cres = {}

    def _finalize_signature(self, cres, args, return_type):
        '''Slated for deprecation, use ufuncbuilder._finalize_ufunc_signature()
        instead.
        '''
        return _finalize_ufunc_signature(cres, args, return_type)

    def build_ufunc(self):
        with global_compiler_lock:
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
                if PYVERSION >= (3, 0):
                    argspec = inspect.getfullargspec(self.py_func)
                else:
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
            ufunc = _internal.fromfunc(
                self.py_func.__name__, self.py_func.__doc__,
                ptrlist, dtypelist, inct, outct, datlist,
                keepalive, self.identity,
            )

            return ufunc

    def build(self, cres, signature):
        '''Slated for deprecation, use
        ufuncbuilder._build_element_wise_ufunc_wrapper().
        '''
        return _build_element_wise_ufunc_wrapper(cres, signature)


class GUFuncBuilder(_BaseUFuncBuilder):

    # TODO handle scalar
    def __init__(self, py_func, signature, identity=None, cache=False,
                 targetoptions={}):
        self.py_func = py_func
        self.identity = parse_identity(identity)
        self.nb_func = jit(target='npyufunc', cache=cache)(py_func)
        self.signature = signature
        self.sin, self.sout = parse_signature(signature)
        self.targetoptions = targetoptions
        self.cache = cache
        self._sigs = []
        self._cres = {}

    def _finalize_signature(self, cres, args, return_type):
        if not cres.objectmode and cres.signature.return_type != types.void:
            raise TypeError("gufunc kernel must have void return type")

        if return_type is None:
            return_type = types.void

        return return_type(*args)

    @global_compiler_lock
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
        ufunc = _internal.fromfunc(
            self.py_func.__name__, self.py_func.__doc__,
            ptrlist, dtypelist, inct, outct, datlist,
            keepalive, self.identity, self.signature,
        )
        return ufunc

    def build(self, cres):
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """
        # Buider wrapper for ufunc entry point
        signature = cres.signature
        info = build_gufunc_wrapper(
            self.py_func, cres, self.sin, self.sout,
            cache=self.cache, is_parfors=False,
        )

        env = info.env
        ptr = info.library.get_pointer_to_function(info.name)
        # Get dtypes
        dtypenums = []
        for a in signature.args:
            if isinstance(a, types.Array):
                ty = a.dtype
            else:
                ty = a
            dtypenums.append(as_dtype(ty).num)
        return dtypenums, ptr, env
