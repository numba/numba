from numba import typeof
from numba.core import types
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba.np.ufunc.sigparse import parse_signature
from numba.np.numpy_support import ufunc_find_matching_loop
from numba.core import serialize


class GUFunc(serialize.ReduceMixin):
    """
    Dynamic generalized universal function (GUFunc)
    intended to act like a normal Numpy gufunc, but capable
    of call-time (just-in-time) compilation of fast loops
    specialized to inputs.
    """

    def __init__(self, py_func, signature, identity=None, cache=None,
                 is_dynamic=False, targetoptions={}):
        self.ufunc = None
        self._frozen = False
        self._is_dynamic = is_dynamic
        self._identity = identity

        # GUFunc cannot inherit from GUFuncBuilder because "identity"
        # is a property of GUFunc. Thus, we hold a reference to a GUFuncBuilder
        # object here
        self.gufunc_builder = GUFuncBuilder(
            py_func, signature, identity, cache, targetoptions)

    def _reduce_states(self):
        gb = self.gufunc_builder
        dct = dict(
            py_func=gb.py_func,
            signature=gb.signature,
            identity=self._identity,
            cache=gb.cache,
            is_dynamic=self._is_dynamic,
            targetoptions=gb.targetoptions,
            typesigs=gb._sigs,
            frozen=self._frozen,
        )
        return dct

    @classmethod
    def _rebuild(cls, py_func, signature, identity, cache, is_dynamic,
                 targetoptions, typesigs, frozen):
        self = cls(py_func=py_func, signature=signature, identity=identity,
                   cache=cache, is_dynamic=is_dynamic,
                   targetoptions=targetoptions)
        for sig in typesigs:
            self.add(sig)
        self.build_ufunc()
        self._frozen = frozen
        return self

    def __repr__(self):
        return f"<numba._GUFunc '{self.__name__}'>"

    def add(self, fty):
        self.gufunc_builder.add(fty)

    def build_ufunc(self):
        self.ufunc = self.gufunc_builder.build_ufunc()
        return self

    def disable_compile(self):
        """
        Disable the compilation of new signatures at call time.
        """
        # If disabling compilation then there must be at least one signature
        assert len(self.gufunc_builder._sigs) > 0
        self._frozen = True

    @property
    def is_dynamic(self):
        return self._is_dynamic

    @property
    def __doc__(self):
        return self.ufunc.__doc__

    @property
    def __name__(self):
        return self.gufunc_builder.py_func.__name__

    @property
    def nin(self):
        return self.ufunc.nin

    @property
    def nout(self):
        return self.ufunc.nout

    @property
    def nargs(self):
        return self.ufunc.nargs

    @property
    def ntypes(self):
        return self.ufunc.ntypes

    @property
    def types(self):
        return self.ufunc.types

    @property
    def identity(self):
        return self.ufunc.identity

    def _get_ewise_dtypes(self, args):
        argtys = map(lambda x: typeof(x), args)
        tys = []
        for argty in argtys:
            if isinstance(argty, types.Array):
                tys.append(argty.dtype)
            else:
                tys.append(argty)
        return tys

    def _num_args_match(self, *args):
        parsed_sig = parse_signature(self.gufunc_builder.signature)
        # parsed_sig[1] has always length 1
        return len(args) == len(parsed_sig[0]) + 1

    def _get_signature(self, *args):
        parsed_sig = parse_signature(self.gufunc_builder.signature)
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

        # add return type to signature
        retty = ewise_types[-1]
        ret_ndim = len(parsed_sig[-1][0]) or 1  # small hack to return scalar
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
            sig = self._get_signature(*args)
            self.add(sig)
            self.build_ufunc()
        return self.ufunc(*args, **kwargs)
