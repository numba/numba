from __future__ import absolute_import, print_function

import random

import numpy as np

from .. import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
                        Registry, signature)


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

# Basics

@infer_global(random.getrandbits, typing_key="random.getrandbits")
class Random_getrandbits(ConcreteTemplate):
    cases = [signature(types.uint64, types.int32)]

@infer_global(random.random, typing_key="random.random")
@infer_global(np.random.random, typing_key="np.random.random")
class Random_random(ConcreteTemplate):
    cases = [signature(types.float64)]

@infer_global(random.randint, typing_key="random.randint")
class Random_randint(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _int_types]

@infer_global(np.random.randint, typing_key="np.random.randint")
class Random_randint(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp) for tp in _int_types]

@infer_global(random.randrange, typing_key="random.randrange")
class Random_randrange(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp) for tp in _int_types]
    cases += [signature(tp, tp, tp, tp) for tp in _int_types]

@infer_global(random.seed, typing_key="random.seed")
@infer_global(np.random.seed, typing_key="np.random.seed")
class Random_seed(ConcreteTemplate):
    cases = [signature(types.void, types.uint32)]

# Distributions

@infer_global(np.random.geometric, typing_key="np.random.geometric")
@infer_global(np.random.logseries, typing_key="np.random.logseries")
@infer_global(np.random.zipf, typing_key="np.random.zipf")
class Numpy_geometric(ConcreteTemplate):
    cases = [signature(types.int64, tp) for tp in _float_types]

@infer_global(np.random.binomial, typing_key="np.random.binomial")
@infer_global(np.random.negative_binomial,
                          typing_key="np.random.negative_binomial")
class Numpy_negative_binomial(ConcreteTemplate):
    cases = [signature(types.int64, types.int64, tp) for tp in _float_types]

@infer_global(np.random.poisson, typing_key="np.random.poisson")
class Numpy_poisson(ConcreteTemplate):
    cases = [signature(types.int64, tp) for tp in _float_types]
    cases += [signature(types.int64)]

@infer_global(np.random.exponential, typing_key="np.random.exponential")
@infer_global(np.random.rayleigh, typing_key="np.random.rayleigh")
class Numpy_exponential(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in _float_types]
    cases += [signature(tp) for tp in _float_types]

@infer_global(np.random.hypergeometric, typing_key="np.random.hypergeometric")
class Numpy_hypergeometric(ConcreteTemplate):
    cases = [signature(tp, tp, tp, tp) for tp in _int_types]

@infer_global(np.random.laplace, typing_key="np.random.laplace")
@infer_global(np.random.logistic, typing_key="np.random.logistic")
@infer_global(np.random.lognormal, typing_key="np.random.lognormal")
@infer_global(np.random.normal, typing_key="np.random.normal")
class Numpy_normal(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
    cases += [signature(tp, tp) for tp in _float_types]
    cases += [signature(tp) for tp in _float_types]

@infer_global(np.random.gamma, typing_key="np.random.gamma")
class Numpy_gamma(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
    cases += [signature(tp, tp) for tp in _float_types]

@infer_global(np.random.triangular, typing_key="np.random.triangular")
class Random_ternary_distribution(ConcreteTemplate):
    cases = [signature(tp, tp, tp, tp) for tp in _float_types]

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
class Random_binary_distribution(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]

@infer_global(np.random.chisquare, typing_key="np.random.chisquare")
@infer_global(np.random.pareto, typing_key="np.random.pareto")
@infer_global(np.random.power, typing_key="np.random.power")
@infer_global(np.random.standard_gamma, typing_key="np.random.standard_gamma")
@infer_global(np.random.standard_t, typing_key="np.random.standard_t")
@infer_global(np.random.weibull, typing_key="np.random.weibull")
@infer_global(random.expovariate, typing_key="random.expovariate")
@infer_global(random.paretovariate, typing_key="random.paretovariate")
class Random_unary_distribution(ConcreteTemplate):
    cases = [signature(tp, tp) for tp in _float_types]

@infer_global(np.random.standard_cauchy,
                          typing_key="np.random.standard_cauchy")
@infer_global(np.random.standard_normal,
                          typing_key="np.random.standard_normal")
@infer_global(np.random.standard_exponential,
                          typing_key="np.random.standard_exponential")
@infer_global(np.random.rand, typing_key="np.random.rand")
@infer_global(np.random.randn, typing_key="np.random.randn")
class Random_nullary_distribution(ConcreteTemplate):
    cases = [signature(tp) for tp in _float_types]

@infer_global(random.triangular, typing_key="random.triangular")
class Random_triangular(ConcreteTemplate):
    cases = [signature(tp, tp, tp) for tp in _float_types]
    cases += [signature(tp, tp, tp, tp) for tp in _float_types]

# Other

@infer_global(random.shuffle, typing_key="random.shuffle")
@infer_global(np.random.shuffle, typing_key="np.random.shuffle")
class Random_shuffle(AbstractTemplate):
    def generic(self, args, kws):
        arr, = args
        if isinstance(arr, types.Buffer) and arr.ndim == 1 and arr.mutable:
            return signature(types.void, arr)
