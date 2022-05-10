import numpy as np
import numba
import pytest

from numpy.random import MT19937, Generator


class TestRandomGenerators:
    def check_numpy_parity(self, distribution_func,
                           bitgen_instance=None, seed=1):
        distribution_func = numba.njit(distribution_func)
        if bitgen_instance is None:
            numba_rng_instance = np.random.default_rng(seed=seed)
            numpy_rng_instance = np.random.default_rng(seed=seed)
        else:
            numba_rng_instance = Generator(bitgen_instance(seed))
            numpy_rng_instance = Generator(bitgen_instance(seed))

        for size in [None, (), (100,), (10,20,30)]:
            numba_res = distribution_func(numba_rng_instance, size)
            numpy_res = distribution_func.py_func(numpy_rng_instance, size)

            assert np.allclose(numba_res, numpy_res)

    @pytest.mark.parametrize("dtype", [
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.bool_
    ])
    def test_integers(self, dtype):
        dist_func = lambda x, size:x.integers(1, size=size, dtype=dtype)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_choice(self):
        dist_func = lambda x, size:x.choice(100, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_normal(self):
        dist_func = lambda x, size:x.standard_normal(size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_exponential(self):
        dist_func = lambda x, size:x.standard_exponential(size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_gamma(self):
        dist_func = lambda x, size:x.standard_gamma(shape=3.0, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_normal(self):
        dist_func = lambda x, size:x.normal(loc=1.5, scale=3, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_exponential(self):
        dist_func = lambda x, size:x.exponential(scale=1.5,size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_gamma(self):
        dist_func = lambda x, size:x.gamma(shape=5.0, scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_beta(self):
        dist_func = lambda x, size:x.beta(a=1.5, b=2.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_f(self):
        dist_func = lambda x, size:x.f(dfnum=2, dfden=3, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_chisquare(self):
        dist_func = lambda x, size:x.chisquare(df=2, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_cauchy(self):
        dist_func = lambda x, size:x.standard_cauchy(size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_pareto(self):
        dist_func = lambda x, size:x.pareto(a=1.0, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_weibull(self):
        dist_func = lambda x, size:x.weibull(a=1.0, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_power(self):
        dist_func = lambda x, size:x.power(a=0.75, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_laplace(self):
        dist_func = lambda x, size:x.laplace(loc=1.0,scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_gumbel(self):
        dist_func = lambda x, size:x.gumbel(loc=1.0,scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_logistic(self):
        dist_func = lambda x, size:x.logistic(loc=1.0,scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_lognormal(self):
        dist_func = lambda x, size:x.lognormal(mean=5.0, sigma=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_rayleigh(self):
        dist_func = lambda x, size:x.rayleigh(scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_standard_t(self):
        dist_func = lambda x, size:x.standard_t(df=2, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_wald(self):
        dist_func = lambda x, size:x.wald(mean=5.0, scale=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_vonmises(self):
        dist_func = lambda x, size:x.vonmises(mu=5.0, kappa=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_geometric(self):
        dist_func = lambda x, size:x.geometric(p=0.75, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_zipf(self):
        dist_func = lambda x, size:x.zipf(a=1.5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_triangular(self):
        dist_func = lambda x, size:x.triangular(left=0, mode=3,
                                                right=5, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_poisson(self):
        # TODO: Find why this functionality fails
        # Probably float/double to int datatype conversion
        dist_func = lambda x, size:x.poisson(lam=15, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)

    def test_negative_binomial(self):
        dist_func = lambda x, size:x.negative_binomial(n=1, p=0.1, size=size)
        self.check_numpy_parity(dist_func)
        self.check_numpy_parity(dist_func, bitgen_instance=MT19937)
