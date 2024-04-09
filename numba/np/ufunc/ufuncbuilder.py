# -*- coding: utf-8 -*-

import functools
import inspect
import warnings
from contextlib import contextmanager

from numba.core import config, targetconfig, errors
from numba.core.decorators import jit
from numba.core.descriptors import TargetDescriptor
from numba.core.extending import is_jitted
from numba.core.options import TargetOptions, include_default_options
from numba.core.typing import templates, npydecl, signature, typeof
from numba.core.registry import cpu_target
from numba.core.target_extension import dispatcher_registry, target_registry
from numba.core import utils, types, serialize, compiler, sigutils
from numba.np.numpy_support import as_dtype, ufunc_find_matching_loop
from numba.np.ufunc import _internal
from numba.np.ufunc.ufunc_base import UfuncBase, UfuncLowererBase
from numba.np.ufunc.sigparse import parse_signature
from numba.np.ufunc.wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.core.caching import FunctionCache, NullCache
from numba.core.compiler_lock import global_compiler_lock


_options_mixin = include_default_options(
    "nopython",
    "forceobj",
    "boundscheck",
    "fastmath",
    "target_backend",
    "writable_args"
)


class UFuncTargetOptions(_options_mixin, TargetOptions):

    def finalize(self, flags, options):
        if not flags.is_set("enable_pyobject"):
            flags.enable_pyobject = True

        if not flags.is_set("enable_looplift"):
            flags.enable_looplift = True

        flags.inherit_if_not_set("nrt", default=True)

        if not flags.is_set("debuginfo"):
            flags.debuginfo = config.DEBUGINFO_DEFAULT

        if not flags.is_set("boundscheck"):
            flags.boundscheck = flags.debuginfo

        flags.enable_pyobject_looplift = True

        flags.inherit_if_not_set("fastmath")


class UFuncTarget(TargetDescriptor):
    options = UFuncTargetOptions

    def __init__(self):
        super().__init__('ufunc')

    @property
    def typing_context(self):
        return cpu_target.typing_context

    @property
    def target_context(self):
        return cpu_target.target_context


ufunc_target = UFuncTarget()


class UFuncDispatcher(serialize.ReduceMixin):
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

    def _reduce_states(self):
        """
        NOTE: part of ReduceMixin protocol
        """
        return dict(
            pyfunc=self.py_func,
            locals=self.locals,
            targetoptions=self.targetoptions,
        )

    @classmethod
    def _rebuild(cls, pyfunc, locals, targetoptions):
        """
        NOTE: part of ReduceMixin protocol
        """
        return cls(py_func=pyfunc, locals=locals, targetoptions=targetoptions)

    def enable_caching(self):
        self.cache = FunctionCache(self.py_func)

    def compile(self, sig, locals={}, **targetoptions):
        locs = self.locals.copy()
        locs.update(locals)

        topt = self.targetoptions.copy()
        topt.update(targetoptions)

        flags = compiler.Flags()
        self.targetdescr.options.parse_as_flags(flags, topt)

        flags.no_cpython_wrapper = True
        flags.error_model = "numpy"
        # Disable loop lifting
        # The feature requires a real
        #  python function
        flags.enable_looplift = False

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
            with targetconfig.ConfigStack().enter(flags.copy()):
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


dispatcher_registry[target_registry['npyufunc']] = UFuncDispatcher


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


@contextmanager
def _suppress_deprecation_warning_nopython_not_supplied():
    """This suppresses the NumbaDeprecationWarning that occurs through the use
    of `jit` without the `nopython` kwarg. This use of `jit` occurs in a few
    places in the `{g,}ufunc` mechanism in Numba, predominantly to wrap the
    "kernel" function."""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                category=errors.NumbaDeprecationWarning,
                                message=(".*The 'nopython' keyword argument "
                                         "was not supplied*"),)
        yield


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


def make_gufunc_kernel(_dufunc):
    from numba.np import npyimpl

    class GUFuncKernel(npyimpl._Kernel):
        """
        npyimpl._Kernel subclass responsible for lowering a gufunc kernel
        (element-wise function) inside a broadcast loop (which is
        generated by npyimpl.numpy_gufunc_kernel()).
        """
        dufunc = _dufunc

        def __init__(self, context, builder, outer_sig):
            super().__init__(context, builder, outer_sig)
            ewise_types = self.dufunc._get_ewise_dtypes(outer_sig.args)
            self.inner_sig, self.cres = self.dufunc.find_ewise_function(
                ewise_types)

        def cast(self, val, fromty, toty):
            # Handle the case where "fromty" is an array and "toty" a scalar
            if isinstance(fromty, types.Array) and not \
                    isinstance(toty, types.Array):
                return super().cast(val, fromty.dtype, toty)
            return super().cast(val, fromty, toty)

        def generate(self, *args):
            if self.cres.objectmode:
                msg = ('Calling a guvectorize function in object mode is not '
                       'supported yet.')
                raise errors.NumbaRuntimeError(msg)
            self.context.add_linking_libs((self.cres.library,))
            return super().generate(*args)

    GUFuncKernel.__name__ += _dufunc.__name__
    return GUFuncKernel


class GUFuncLowerer(UfuncLowererBase):
    '''Callable class responsible for lowering calls to a specific gufunc.
    '''
    def __init__(self, gufunc):
        from numba.np import npyimpl
        super().__init__(gufunc,
                         make_gufunc_kernel,
                         npyimpl.numpy_gufunc_kernel)


class UFuncBuilder(_BaseUFuncBuilder):

    def __init__(self, py_func, identity=None, cache=False, targetoptions={}):
        if is_jitted(py_func):
            py_func = py_func.py_func
        self.py_func = py_func
        self.identity = parse_identity(identity)
        with _suppress_deprecation_warning_nopython_not_supplied():
            self.nb_func = jit(_target='npyufunc',
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
                ptrlist.append(int(ptr))
                keepalive.append((cres.library, env))

            datlist = [None] * len(ptrlist)

            if cres is None:
                argspec = inspect.getfullargspec(self.py_func)
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


class GUFuncBuilder(serialize.ReduceMixin, _BaseUFuncBuilder, UfuncBase):

    # TODO handle scalar
    def __init__(self, py_func, signature, identity=None, cache=False,
                 is_dynamic=False, targetoptions={}, writable_args=()):
        self.py_func = py_func
        self._identity = parse_identity(identity)
        with _suppress_deprecation_warning_nopython_not_supplied():
            self.nb_func = jit(_target='npyufunc', cache=cache)(py_func)
        self._signature = signature
        self.sin, self.sout = parse_signature(signature)
        self.targetoptions = targetoptions
        self.cache = cache
        self._sigs = []
        self._cres = {}
        self._frozen = False
        self.ufunc = None
        self._is_dynamic = is_dynamic

        transform_arg = _get_transform_arg(py_func)
        self.writable_args = tuple([transform_arg(a) for a in writable_args])
        self.__name__ = self.py_func.__name__
        self.__doc__ = self.py_func.__doc__
        self._dispatcher = self.nb_func
        self._initialize()

        functools.update_wrapper(self, py_func)

    def _initialize(self):
        self.build_ufunc()
        self._install_type()
        self._lower_me = GUFuncLowerer(self)
        self._install_cg()

    def disable_compile(self):
        """
        Disable the compilation of new signatures at call time.
        """
        # If disabling compilation then there must be at least one signature
        assert len(self._dispatcher.overloads) > 0
        self._frozen = True

    def _finalize_signature(self, cres, args, return_type):
        if not cres.objectmode and cres.signature.return_type != types.void:
            raise TypeError("gufunc kernel must have void return type")

        if return_type is None:
            return_type = types.void

        return return_type(*args)

    @global_compiler_lock
    def build_ufunc(self):
        type_list = []
        func_list = []
        if not self.nb_func:
            raise TypeError("No definition")

        # Get signature in the order they are added
        keepalive = []
        for sig in self._sigs:
            cres = self._cres[sig]
            dtypenums, ptr, env = self.build(cres)
            type_list.append(dtypenums)
            func_list.append(int(ptr))
            keepalive.append((cres.library, env))

        datalist = [None] * len(func_list)

        nin = len(self.sin)
        nout = len(self.sout)

        # Pass envs to fromfuncsig to bind to the lifetime of the ufunc object
        ufunc = _internal.fromfunc(
            self.py_func.__name__, self.py_func.__doc__,
            func_list, type_list, nin, nout, datalist,
            keepalive, self._identity, self._signature, self.writable_args
        )
        self.ufunc = ufunc
        return self

    def build(self, cres):
        """
        Returns (dtype numbers, function ptr, EnvironmentObject)
        """
        # Builder wrapper for ufunc entry point
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

    def _reduce_states(self):
        dct = dict(
            py_func=self.py_func,
            signature=self._signature,
            identity=self.identity,
            cache=self.cache,
            is_dynamic=self._is_dynamic,
            targetoptions=self.targetoptions,
            writable_args=self.writable_args,
            typesigs=self._sigs,
            frozen=self._frozen,
        )
        return dct

    @classmethod
    def _rebuild(cls, py_func, signature, identity, cache, is_dynamic,
                 targetoptions, writable_args, typesigs, frozen):
        self = cls(py_func=py_func, signature=signature, identity=identity,
                   cache=cache, is_dynamic=is_dynamic,
                   targetoptions=targetoptions, writable_args=writable_args)
        for sig in typesigs:
            self.add(sig)
        self.build_ufunc()
        self._frozen = frozen
        return self

    def __repr__(self):
        return f"<numba._GUFunc '{self.__name__}'>"

    @property
    def is_dynamic(self):
        return self._is_dynamic

    def _install_type(self, typingctx=None):
        """Constructs and installs a typing class for a gufunc object in the
        input typing context.  If no typing context is given, then
        _install_type() installs into the typing context of the
        dispatcher object (should be same default context used by
        jit() and njit()).
        """
        if typingctx is None:
            typingctx = self._dispatcher.targetdescr.typing_context
        _ty_cls = type('GUFuncTyping_' + self.__name__,
                       (templates.AbstractTemplate,),
                       dict(key=self, generic=self._type_me))
        typingctx.insert_user_function(self, _ty_cls)

    def expected_ndims(self):
        parsed_sig = parse_signature(self._signature)
        return (tuple(map(len, parsed_sig[0])), tuple(map(len, parsed_sig[1])))

    def _type_me(self, argtys, kws):
        """
        Implement AbstractTemplate.generic() for the typing class
        built by gufunc._install_type().

        Return the call-site signature after either validating the
        element-wise signature or compiling for it.
        """
        assert not kws
        ufunc = self.ufunc
        sig = self.signature
        inp_ndims, out_ndims = self.expected_ndims()
        ndims = inp_ndims + out_ndims

        assert len(argtys), len(ndims)
        for idx, arg in enumerate(argtys):
            if isinstance(arg, types.Array) and arg.ndim < ndims[idx]:
                kind = "Input" if idx < len(inp_ndims) else "Output"
                i = idx if idx < len(inp_ndims) else idx - len(inp_ndims)
                msg = (
                    f"{self.__name__}: {kind} operand {i} does not have "
                    f"enough dimensions (has {arg.ndim}, gufunc core with "
                    f"signature {sig} requires {ndims[idx]})")
                raise errors.NumbaTypeError(msg)

        _handle_inputs_result = npydecl.Numpy_rules_ufunc._handle_inputs(
            ufunc, argtys, kws)
        ewise_types, _, _, _ = _handle_inputs_result
        sig, _ = self.find_ewise_function(ewise_types)

        if sig is None:
            # Matching element-wise signature was not found; must
            # compile.
            if self._frozen:
                msg = f"cannot call {self} with types {argtys}"
                raise errors.NumbaTypeError(msg)
            self._compile_for_argtys(ewise_types)
            # double check to ensure there is a match
            sig, _ = self.find_ewise_function(ewise_types)
            if sig == (None, None):
                msg = f"Fail to compile {self.__name__} with types {argtys}"
                raise errors.NumbaTypeError(msg)

            assert sig is not None

        return signature(types.none, *argtys)

    def _compile_for_argtys(self, argtys, return_type=None):
        # Compile a new guvectorize function! Use the gufunc signature
        # i.e. (n,m),(m)->(n)
        # plus ewise_types to build a numba function type
        fnty = self._get_function_type(*argtys)
        self.add(fnty)
        self.build_ufunc()

    def match_signature(self, ewise_types, sig):
        dtypes = self._get_ewise_dtypes(sig.args)
        return tuple(dtypes) == tuple(ewise_types)

    def _get_ewise_dtypes(self, args):
        argtys = map(lambda arg: arg if isinstance(arg, types.Type) else
                     typeof.typeof(arg), args)
        tys = []
        for argty in argtys:
            if isinstance(argty, types.Array):
                tys.append(argty.dtype)
            else:
                tys.append(argty)
        return tys

    def _num_args_match(self, *args):
        parsed_sig = parse_signature(self._signature)
        return len(args) == len(parsed_sig[0]) + len(parsed_sig[1])

    def _get_function_type(self, *args):
        parsed_sig = parse_signature(self._signature)
        # ewise_types is a list of [int32, int32, int32, ...]
        ewise_types = self._get_ewise_dtypes(args)

        # first time calling the gufunc
        # generate a signature based on input arguments
        l = []
        for idx, sig_dim in enumerate(parsed_sig[0]):
            ndim = len(sig_dim)
            if ndim == 0:  # append scalar
                l.append(ewise_types[idx])
            else:
                l.append(types.Array(ewise_types[idx], ndim, 'A'))

        offset = len(parsed_sig[0])
        # add return type to signature
        for idx, sig_dim in enumerate(parsed_sig[1]):
            retty = ewise_types[idx + offset]
            ret_ndim = len(sig_dim) or 1  # small hack to return scalars
            l.append(types.Array(retty, ret_ndim, 'A'))

        return types.none(*l)

    def __call__(self, *args, **kwargs):
        # If compilation is disabled OR it is NOT a dynamic gufunc
        # call the underlying gufunc
        if self._frozen or not self.is_dynamic:
            return self.ufunc(*args, **kwargs)
        elif "out" in kwargs:
            # If "out" argument is supplied
            args += (kwargs.pop("out"),)

        if self._num_args_match(*args) is False:
            # It is not allowed to call a dynamic gufunc without
            # providing all the arguments
            # see: https://github.com/numba/numba/pull/5938#discussion_r506429392  # noqa: E501
            msg = (
                f"Too few arguments for function '{self.__name__}'. "
                "Note that the pattern `out = gufunc(Arg1, Arg2, ..., ArgN)` "
                "is not allowed. Use `gufunc(Arg1, Arg2, ..., ArgN, out) "
                "instead.")
            raise TypeError(msg)

        # at this point we know the gufunc is a dynamic one
        ewise = self._get_ewise_dtypes(args)
        if not (self.ufunc and ufunc_find_matching_loop(self.ufunc, ewise)):
            # A previous call (@njit -> @guvectorize) may have compiled a
            # version for the element-wise dtypes. In this case, we don't need
            # to compile it again, just build the (g)ufunc
            if not self.find_ewise_function(ewise) != (None, None):
                sig = self._get_function_type(*args)
                self.add(sig)
                self.build_ufunc()

        return self.ufunc(*args, **kwargs)


def _get_transform_arg(py_func):
    """Return function that transform arg into index"""
    args = inspect.getfullargspec(py_func).args
    pos_by_arg = {arg: i for i, arg in enumerate(args)}

    def transform_arg(arg):
        if isinstance(arg, int):
            return arg

        try:
            return pos_by_arg[arg]
        except KeyError:
            msg = (f"Specified writable arg {arg} not found in arg list "
                   f"{args} for function {py_func.__qualname__}")
            raise RuntimeError(msg)

    return transform_arg
