from __future__ import absolute_import, print_function

import random

import numpy as np

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        CallableTemplate, Registry, signature)
from ..numpy_support import version as np_version


registry = Registry()
infer = registry.register
infer_global = registry.register_global
infer_getattr = registry.register_attr


# random.random(), random.seed() etc. are not plain functions, they are bound
# methods of a private object.  We have to be careful to use a well-known
# object (e.g. the string "random.seed") as a key, not the bound method itself.
# (same for np.random.random(), etc.)

_int_types = sorted(set((types.intp, types.int64)))
# Should we support float32?
_float_types = [types.float64]


#
# Basics
#

def normalize_shape(shape):
    if isinstance(shape, types.Integer):
        return types.intp, 1
    elif (isinstance(shape, types.BaseTuple) and
          all(isinstance(v, types.Integer)) for v in shape):
        ndim = len(shape)
        return types.UniTuple(types.intp, ndim), ndim
    else:
        raise TypeError("invalid size type %s" % (shape,))


class RandomTemplate(CallableTemplate):
    """
    A template helper to transparently handle the typing of array-returning
    np.random.* functions.
    """

    def array_typer(self, scalar_typer, size=None):
        prefix = self.key.split('.')[0]
        assert prefix in ('np', 'random'), self.key

        if size is None:
            # Scalar variant
            def typer(*args, **kwargs):
                return scalar_typer(*args, **kwargs)
        else:
            # Array variant (only for the 'np.random.*' namespace)
            def typer(*args, **kwargs):
                if prefix == 'random':
                    raise TypeError("unexpected size parameter for %r"
                                    % (self.key,))
                shape, ndim = normalize_shape(size)
                # Type the scalar variant and wrap the result in an array
                # of the appropriate dimensionality.
                sig = scalar_typer(*args, **kwargs)
                if sig is not None:
                    return signature(
                        types.Array(sig.return_type, ndim, 'C'),
                        *(sig.args + (shape,)))

        return typer


class ConcreteRandomTemplate(RandomTemplate):
    """
    A RandomTemplate subclass using the `cases` attribute as a list of
    allowed scalar signatures.
    """

    def array_typer(self, size=None):
        key = self.key
        cases = self.cases
        context = self.context

        def concrete_scalar_typer(*args, **kwargs):
            # Filter out omitted args
            while args and args[-1] is None:
                args = args[:-1]
            return context.resolve_overload(key, cases, args, kwargs)

        return RandomTemplate.array_typer(self, concrete_scalar_typer, size)


@infer_global(random.getrandbits, typing_key="random.getrandbits")
class Random_getrandbits(ConcreteTemplate):
    cases = [signature(types.uint64, types.int32)]

@infer_global(random.random, typing_key="random.random")
@infer_global(np.random.random, typing_key="np.random.random")
class Random_random(ConcreteRandomTemplate):
    cases = [signature(types.float64)]

    def generic(self):
        def typer(size=None):
            return self.array_typer(size)()
        return typer

if np_version >= (1, 17):
    infer_global(
        np.random.random_sample,
        typing_key="np.random.random_sample",
    )(Random_random)
    infer_global(
        np.random.sample,
        typing_key="np.random.sample",
    )(Random_random)
    infer_global(
        np.random.ranf,
        typing_key="np.random.ranf",
    )(Random_random)


@infer_global(random.randint, typing_key="random.randint")
class Random_randint(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _int_types]

@infer_global(np.random.randint, typing_key="np.random.randint")
class Random_randint(ConcreteRandomTemplate):
    cases = [signature(tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp) for tp in _int_types]

    def generic(self):
        def typer(low, high=None, size=None):
            return self.array_typer(size)(low, high)
        return typer


@infer_global(random.randrange, typing_key="random.randrange")
class Random_randrange(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp, tp) for tp in _int_types]

@infer_global(random.seed, typing_key="random.seed")
@infer_global(np.random.seed, typing_key="np.random.seed")
class Random_seed(ConcreteTemplate):
    cases = [signature(types.void, types.uint32)]


#
# Distributions
#

@infer_global(np.random.geometric, typing_key="np.random.geometric")
@infer_global(np.random.logseries, typing_key="np.random.logseries")
@infer_global(np.random.zipf, typing_key="np.random.zipf")
class Numpy_geometric(ConcreteRandomTemplate):
    cases = [signature(types.int64, tp) for tp in _float_types]

    def generic(self):
        def typer(a, size=None):
            return self.array_typer(size)(a)
        return typer

@infer_global(np.random.binomial, typing_key="np.random.binomial")
@infer_global(np.random.negative_binomial,
                          typing_key="np.random.negative_binomial")
class Numpy_negative_binomial(ConcreteRandomTemplate):
    cases = [signature(types.int64, types.int64, tp) for tp in _float_types]

    def generic(self):
        def typer(n, p, size=None):
            return self.array_typer(size)(n, p)
        return typer

@infer_global(np.random.poisson, typing_key="np.random.poisson")
class Numpy_poisson(ConcreteRandomTemplate):
    cases = [signature(types.int64, tp) for tp in _float_types]
    cases += [signature(types.int64)]

    def generic(self):
        def typer(lam=None, size=None):
            return self.array_typer(size)(lam)
        return typer

@infer_global(np.random.exponential, typing_key="np.random.exponential")
@infer_global(np.random.rayleigh, typing_key="np.random.rayleigh")
class Numpy_exponential(ConcreteRandomTemplate):
    cases = [signature(tp, tp) for tp in _float_types]
    cases += [signature(tp) for tp in _float_types]

    def generic(self):
        def typer(scale=None, size=None):
            return self.array_typer(size)(scale)
        return typer

@infer_global(np.random.hypergeometric, typing_key="np.random.hypergeometric")
class Numpy_hypergeometric(ConcreteRandomTemplate):
    cases = [signature(tp, tp, tp, tp) for tp in _int_types]

    def generic(self):
        def typer(ngood, nbad, nsample, size=None):
            return self.array_typer(size)(ngood, nbad, nsample)
        return typer

@infer_global(np.random.laplace, typing_key="np.random.laplace")
@infer_global(np.random.logistic, typing_key="np.random.logistic")
@infer_global(np.random.lognormal, typing_key="np.random.lognormal")
@infer_global(np.random.normal, typing_key="np.random.normal")
class Numpy_normal(ConcreteRandomTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
    cases += [signature(tp, tp) for tp in _float_types]
    cases += [signature(tp) for tp in _float_types]

    def generic(self):
        def typer(loc=None, scale=None, size=None):
            return self.array_typer(size)(loc, scale)
        return typer

@infer_global(np.random.gamma, typing_key="np.random.gamma")
class Numpy_gamma(ConcreteRandomTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
    cases += [signature(tp, tp) for tp in _float_types]

    def generic(self):
        def typer(shape, scale=None, size=None):
            return self.array_typer(size)(shape, scale)
        return typer

@infer_global(np.random.triangular, typing_key="np.random.triangular")
class Random_ternary_distribution(ConcreteRandomTemplate):
    cases = [signature(tp, tp, tp, tp) for tp in _float_types]

    def generic(self):
        def typer(left, mode, right, size=None):
            return self.array_typer(size)(left, mode, right)
        return typer


@infer_global(np.random.beta, typing_key="np.random.beta")
@infer_global(np.random.f, typing_key="np.random.f")
@infer_global(np.random.gumbel, typing_key="np.random.gumbel")
@infer_global(np.random.uniform, typing_key="np.random.uniform")
@infer_global(np.random.vonmises, typing_key="np.random.vonmises")
@infer_global(np.random.wald, typing_key="np.random.wald")
@infer_global(random.betavariate, typing_key="random.betavariate")
@infer_global(random.gammavariate, typing_key="random.gammavariate")
@infer_global(random.gauss, typing_key="random.gauss")
@infer_global(random.lognormvariate, typing_key="random.lognormvariate")
@infer_global(random.normalvariate, typing_key="random.normalvariate")
@infer_global(random.uniform, typing_key="random.uniform")
@infer_global(random.vonmisesvariate, typing_key="random.vonmisesvariate")
@infer_global(random.weibullvariate, typing_key="random.weibullvariate")
class Random_binary_distribution(ConcreteRandomTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]

    def generic(self):
        def typer(a, b, size=None):
            return self.array_typer(size)(a, b)
        return typer


@infer_global(np.random.chisquare, typing_key="np.random.chisquare")
@infer_global(np.random.pareto, typing_key="np.random.pareto")
@infer_global(np.random.power, typing_key="np.random.power")
@infer_global(np.random.standard_gamma, typing_key="np.random.standard_gamma")
@infer_global(np.random.standard_t, typing_key="np.random.standard_t")
@infer_global(np.random.weibull, typing_key="np.random.weibull")
@infer_global(random.expovariate, typing_key="random.expovariate")
@infer_global(random.paretovariate, typing_key="random.paretovariate")
class Random_unary_distribution(ConcreteRandomTemplate):
    cases = [signature(tp, tp) for tp in _float_types]

    def generic(self):
        def typer(a, size=None):
            return self.array_typer(size)(a)
        return typer


@infer_global(np.random.standard_cauchy,
                          typing_key="np.random.standard_cauchy")
@infer_global(np.random.standard_normal,
                          typing_key="np.random.standard_normal")
@infer_global(np.random.standard_exponential,
                          typing_key="np.random.standard_exponential")
class Random_nullary_distribution(ConcreteRandomTemplate):
    cases = [signature(tp) for tp in _float_types]

    def generic(self):
        def typer(size=None):
            return self.array_typer(size)()
        return typer


@infer_global(random.triangular, typing_key="random.triangular")
class Random_triangular(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
    cases += [signature(tp, tp, tp, tp) for tp in _float_types]

# NOTE: some functions can have @overloads in numba.targets.randomimpl,
# and therefore don't need a typing declaration here.
