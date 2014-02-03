# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import warnings
import numpy as np
from numba.decorators import jit
from numba.targets.registry import target_registry
from numba.targets.descriptors import TargetDescriptor
from numba.targets.options import TargetOptions
from numba import utils, compiler, types, sigutils
from . import _internal
from .sigparse import parse_signature
from .wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.targets import registry


class UFuncTargetOptions(TargetOptions):
    OPTIONS = {
        "nopython" : bool,
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

        if topt.get('nopython', True) == False:
            raise TypeError("nopython option must be False")
        topt['nopython'] = True

        flags = compiler.Flags()
        flags.set("no_compile")
        self.targetdescr.options.parse_as_flags(flags, topt)

        typingctx = self.targetdescr.typing_context
        targetctx = self.targetdescr.target_context

        args, return_type = sigutils.normalize_signature(sig)

        cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                      args=args, return_type=return_type,
                                      flags=flags, locals=locals)

        self.overloads[cres.signature] = cres
        return cres


target_registry['npyufunc'] = UFuncDispatcher


class UFuncBuilder(object):
    def __init__(self, py_func, targetoptions={}):
        self.py_func = py_func
        self.nb_func = jit(target='npyufunc', **targetoptions)(py_func)

    def add(self, sig=None, argtypes=None, restype=None):
        # Handle argtypes
        if argtypes is not None:
            warnings.warn("Keyword argument argtypes is deprecated",
                          DeprecationWarning)
            assert sig is None
            if restype is None:
                sig = tuple(argtypes)
            else:
                sig = restype(*argtypes)
        # Actual work
        self.nb_func.compile(sig)

    def build_ufunc(self):
        dtypelist = []
        ptrlist = []
        if not self.nb_func:
            raise TypeError("No definition")
        for sig, cres in self.nb_func.overloads.items():
            dtypenums, ptr = self.build(cres)
            dtypelist.append(dtypenums)
            ptrlist.append(utils.longint(ptr))
        datlist = [None] * len(ptrlist)

        inct = len(cres.signature.args)
        outct = 1

        # Becareful that fromfunc does not provide full error checking yet.
        # If typenum is out-of-bound, we have nasty memory corruptions.
        # For instance, -1 for typenum will cause segfault.
        # If elements of type-list (2nd arg) is tuple instead,
        # there will also memory corruption. (Seems like code rewrite.)
        ufunc = _internal.fromfunc(ptrlist, dtypelist, inct, outct, datlist)

        return ufunc

    def build(self, cres):
        # Buider wrapper for ufunc entry point
        ctx = cres.target_context
        signature = cres.signature
        wrapper = build_ufunc_wrapper(ctx, cres.llvm_func, signature)
        ctx.engine.add_module(wrapper.module)
        ptr = ctx.engine.get_pointer_to_function(wrapper)
        # Get dtypes
        dtypenums = [np.dtype(a.name).num for a in signature.args]
        dtypenums.append(np.dtype(signature.return_type.name).num)
        return dtypenums, ptr


class GUFuncBuilder(object):
    # TODO handle scalar
    def __init__(self, py_func, signature, targetoptions={}):
        self.py_func = py_func
        self.nb_func = jit(target='npyufunc')(py_func)
        self.signature = signature
        self.sin, self.sout = parse_signature(signature)
        self.targetoptions = targetoptions

    def add(self, sig=None, argtypes=None, restype=None):
        # Handle argtypes
        if argtypes is not None:
            warnings.warn("Keyword argument argtypes is deprecated",
                          DeprecationWarning)
            assert sig is None
            if restype is None:
                sig = tuple(argtypes)
            else:
                sig = restype(*argtypes)
        # Actual work begins
        cres = self.nb_func.compile(sig, **self.targetoptions)
        if cres.signature.return_type != types.void:
            raise TypeError("gufunc kernel must have void return type")

    def build_ufunc(self):
        dtypelist = []
        ptrlist = []
        if not self.nb_func:
            raise TypeError("No definition")

        for sig, cres in self.nb_func.overloads.items():
            dtypenums, ptr = self.build(cres)
            dtypelist.append(dtypenums)
            ptrlist.append(utils.longint(ptr))
        datlist = [None] * len(ptrlist)

        inct = len(self.sin)
        outct = len(self.sout)

        ufunc = _internal.fromfuncsig(ptrlist, dtypelist, inct, outct, datlist,
                                      self.signature)

        return ufunc

    def build(self, cres):
        # Buider wrapper for ufunc entry point
        ctx = cres.target_context
        signature = cres.signature
        wrapper = build_gufunc_wrapper(ctx, cres.llvm_func, signature,
                                       self.sin, self.sout)
        ctx.engine.add_module(wrapper.module)
        ptr = ctx.engine.get_pointer_to_function(wrapper)
        # Get dtypes
        dtypenums = []
        for a in signature.args:
            if isinstance(a, types.Array):
                ty = a.dtype
            else:
                ty = a
            dtypenums.append(np.dtype(ty.name).num)
        return dtypenums, ptr

