# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
from numba.decorators import jit, target_registry
from numba.dispatcher import read_flags, GlobalContext, normalize_signature
from numba import utils, compiler, types
from . import _internal
from .sigparse import parse_signature
from .wrappers import build_ufunc_wrapper, build_gufunc_wrapper


class UFuncDispatcher(object):
    def __init__(self, py_func):
        self.py_func = py_func
        self.overloads = utils.UniqueDict()

    def compile(self, sig, **kws):
        if kws.get("nopython", True) == False:
            raise AssertionError("nopython option must be True")
        if kws.get("forceobj", False) == True:
            raise AssertionError("forceobj option must be False")

        flags = compiler.Flags()
        read_flags(flags, kws)

        glctx = GlobalContext()
        typingctx = glctx.typing_context
        targetctx = glctx.target_context

        args, return_type = normalize_signature(sig)

        cres = compiler.compile_extra(typingctx, targetctx, self.py_func,
                                      args=args, return_type=return_type,
                                      flags=flags)

        self.overloads[cres.signature] = cres
        return cres


target_registry['npyufunc'] = UFuncDispatcher


class UFuncBuilder(object):
    def __init__(self, py_func, kws={}):
        self.py_func = py_func
        self.nb_func = jit(target='npyufunc')(py_func)
        self.kws = kws

    def add(self, sig):
        self.nb_func.compile(sig, nocompile=True, **self.kws)

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
    def __init__(self, py_func, signature, kws={}):
        self.py_func = py_func
        self.nb_func = jit(target='npyufunc')(py_func)
        self.signature = signature
        self.sin, self.sout = parse_signature(signature)
        self.kws = kws

    def add(self, sig):
        cres = self.nb_func.compile(sig, nocompile=True, **self.kws)
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

