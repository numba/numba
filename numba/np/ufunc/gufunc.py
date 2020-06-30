from numba import typeof
from numba.core import types
from numba.np.ufunc.ufuncbuilder import GUFuncBuilder
from numba.np.ufunc.sigparse import parse_signature
from numba.np.numpy_support import ufunc_find_matching_loop


class GUFunc(object):
    """
    Dynamic generalized universal function (GUFunc)
    intended to act like a normal Numpy gufunc, but capable
    of call-time (just-in-time) compilation of fast loops
    specialized to inputs.
    """

    def __init__(self, py_func, signature, identity=None, cache=None,
                 targetoptions={}):
        self.ufunc = None
        self._frozen = False

        # GUFunc cannot inherit from GUFuncBuilder because "identity"
        # is a property of GUFunc. Thus, we hold a reference to a GUFuncBuilder
        # object here
        self.gufunc_builder = GUFuncBuilder(
            py_func, signature, identity, cache, targetoptions)

    def add(self, fty):
        self.gufunc_builder.add(fty)

    def build_ufunc(self):
        self.ufunc = self.gufunc_builder.build_ufunc()
        return self

    def disable_compile(self):
        assert len(self.gufunc_builder._sigs) > 0
        self._frozen = True

    @property
    def __doc__(self):
        return self.ufunc.__doc__

    @property
    def __name__(self):
        return self.ufunc.__name__

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

    def _get_signature(self, *args):
        parsed_sig = parse_signature(self.gufunc_builder.signature)
        ewise_types = self._get_ewise_dtypes(args)  # [int32, int32, int32, ...]  # noqa: E501

        # Two cases here can happen here
        # 1. args contains the output array:
        #      gufunc(A, B, C)
        # 2. args doesn't contains the output array:
        #      C = gufunc(A, B)
        #
        # In the latter case, one would have to guess the type
        # and format (scalar or array). This behavior will be
        # forbidden for now.
        # See: https://github.com/numba/numba/pull/5938#issuecomment-661978921

        # parsed_sig[1] has always length 1
        if len(ewise_types) < len(parsed_sig[0]) + 1:
            msg = (
                f"Too few arguments for function '{self.__name__}'. "
                "Note that the pattern `out = gufunc(Arg1, Arg2, ..., ArgN)` "
                "is not allowed. Use `gufunc(Arg1, Arg2, ..., ArgN, out) instead.")  # noqa: E501
            raise TypeError(msg)

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

    def __call__(self, *args):

        if self._frozen:
            return self.ufunc(*args)

        ewise = self._get_ewise_dtypes(args)
        if not (self.ufunc and ufunc_find_matching_loop(self.ufunc, ewise)):
            sig = self._get_signature(*args)
            self.add(sig)
            self.build_ufunc()
        return self.ufunc(*args)
